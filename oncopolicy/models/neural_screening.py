import torch
import torch.nn as nn
import torch.nn.functional as F
from oncopolicy.models.factory import RegisterModel
from collections import deque
from oncopolicy.utils.generic import to_numpy, to_tensor, sample_preference_vector, NUM_DIM_AUX_FEATURES
from oncopolicy.datasets.personalized_screening import PAD_Y_VALUE
import copy
import numpy as np
import pdb


class AbstractNeuralScreeningModel(nn.Module):
    def __init__(self, args):
        super(AbstractNeuralScreeningModel, self).__init__()
        self.args = args
        self.num_actions =  (self.args.max_screening_interval_in_6mo_counts - self.args.min_screening_interval_in_6mo_counts) + 1
        self.reward_dim = len(self.args.metrics)
        input_dim = NUM_DIM_AUX_FEATURES + args.risk_dimension + self.reward_dim
        if self.args.use_gru_latent_in_screening:
            input_dim += args.prog_hidden_dim
        self.input_dim = input_dim
        self.memory = deque()
        self.priority = deque()
        self.homotopy = 1.0
        self.oracle_epsilon = self.args.max_oracle_prob

    def q(self, x, preference):
        concat_x = torch.cat([x.squeeze(1), preference.squeeze(-1)], dim=-1)
        return self.model(concat_x).view( [-1, self.reward_dim, self.num_actions])

    def target_q(self, x, preference):
        concat_x = torch.cat([x.squeeze(1), preference.squeeze(-1)], dim=-1)
        return self.target(concat_x).view( [-1, self.reward_dim, self.num_actions])

    def reset(self):
        '''
            increase beta for homtopy,
            decrease episolon if decaying
            save model into target
        '''
        self.target = copy.deepcopy(self.model)
        self.homotopy *= self.args.homotopy_decay_rate
        self.oracle_epsilon = max(self.oracle_epsilon * self.args.oracle_decay_rate, self.args.min_oracle_prob)


    def save_transition(self, cur_x, next_x, action, reward_vec, preference, cur_is_censored, next_is_censored, oracle_action):
        assert self.training
        batch_size = cur_x.size()[0]
        _action = action - self.args.min_screening_interval_in_6mo_counts
        _oracle_action = oracle_action - self.args.min_screening_interval_in_6mo_counts
        losses = self.get_loss(cur_x, next_x, _action, reward_vec, next_is_censored, F.relu(_oracle_action)).detach()
        censored = to_numpy(cur_is_censored)
        for idx in range(batch_size):
            if censored[idx] == 1:
                continue
            if self.args.imitate_oracle and to_numpy(_oracle_action[idx]) < 0:
                continue
            self.memory.append( {
                    'state': to_numpy(cur_x[idx]),
                    'next_state': to_numpy(next_x[idx]),
                    'action': to_numpy(_action[idx]),
                    'r': to_numpy(reward_vec[idx]),
                    'terminal': to_numpy(next_is_censored[idx]),
                    'oracle_action': to_numpy(_oracle_action[idx])
                    }
            )
            sample_loss = losses[idx].item()
            assert not np.isnan(sample_loss)
            self.priority.append(sample_loss)

        assert len(self.priority) == len(self.memory)
        for memory in [self.memory, self.priority]:
            while len(memory) > self.args.replay_size:
                memory.popleft()
        assert len(self.priority) == len(self.memory)

    def get_loss(self, cur_x, next_x, action, reward_vec, terminal, oracle_action):
        if hasattr(self.args, 'imitate_oracle') and self.args.imitate_oracle:
            return self.get_imitation_loss(cur_x, oracle_action)
        if self.args.sample_random_preferences:
            return self.get_morl_loss(cur_x, next_x, action, reward_vec, terminal)
        else:
            return self.get_q_learning_loss(cur_x, next_x, action, reward_vec, terminal)

    def get_morl_loss(self, cur_x, next_x, action, reward_vec, terminal):
        B = cur_x.size(0)
        preference = sample_preference_vector(B, True, self.args)
        terminal = terminal.float().unsqueeze(-1)
        # Calc curr estimated value
        cur_q_all_a = self.q(cur_x, preference)
        action_ = action.unsqueeze(-1).expand([B, self.reward_dim, 1]).long()
        cur_q =  cur_q_all_a.gather(index=action_, dim=-1).squeeze(-1)
        # Search envelope for target Q
        _, _, next_morl_q = self.search_envelope(next_x, preference, True, preference)
        # Filter for terminal vs non-terminal values
        non_terminal_value = reward_vec + self.args.reward_decay_lambda * next_morl_q
        terminal_values = reward_vec
        target_q = (terminal * terminal_values) + (1-terminal) * (non_terminal_value)
        # Compute MORL loss functions
        preference = preference.squeeze(-1)
        scalarized_cur_q =  (cur_q * preference).sum(dim=-1)
        scalarized_target_q = (target_q * preference).sum(dim=-1)
        primary_morl_loss = ((target_q - cur_q)**2).sum(dim=-1)
        aux_morl_loss = (scalarized_target_q - scalarized_cur_q)**2
        losses = ((1-self.homotopy) * primary_morl_loss) + (self.homotopy * aux_morl_loss)
        if self.args.envelope_margin_loss_lambda > 0:
            scalarized_env_q, _, _ = self.search_envelope(cur_x, preference, True)
            margin_loss = F.relu( (scalarized_env_q - scalarized_cur_q) + self.args.envelope_margin )
            losses += self.args.envelope_margin_loss_lambda * margin_loss
        return losses

    def get_imitation_loss(self, cur_x, action):
        batch_size = cur_x.size(0)
        preference = sample_preference_vector(batch_size, False, self.args)
        cur_q_all_a = self.q(cur_x, preference)
        scalarized_cur_q =  (cur_q_all_a * preference).sum(dim=1)
        return F.cross_entropy(scalarized_cur_q, action.squeeze(1).long(), reduction='none')

    def get_q_learning_loss(self, cur_x, next_x, action, reward_vec, terminal):
        batch_size = cur_x.size(0)
        preference = sample_preference_vector(batch_size, False, self.args)
        terminal = terminal.float()
        # Calc curr estimated value
        cur_q_all_a = self.q(cur_x, preference)
        action_ = action.unsqueeze(-1).expand([batch_size, self.reward_dim, 1]).long()
        cur_q =  cur_q_all_a.gather(index=action_, dim=-1)
        scalarized_cur_q =  (cur_q * preference).squeeze(-1).sum(dim=-1)
        # Calc bootstrapped q value from target network
        net_reward = (reward_vec * preference.squeeze(-1)).sum(dim=1)
        next_state_q_all_a = (self.target_q(next_x, preference)  * preference).sum(dim=1)
        next_q, _ = torch.max(next_state_q_all_a, 1)
        non_terminal_value = net_reward + self.args.reward_decay_lambda * next_q
        terminal_values = net_reward
        target_q = (terminal * terminal_values) + (1-terminal) * (non_terminal_value)
        losses = (target_q - scalarized_cur_q)**2
        return losses

    def sample(self):
        priority_arr = np.array(self.priority).astype(np.float)
        priority_arr /= priority_arr.sum()
        indicies = np.random.choice( len(self.memory), self.args.batch_size, p=priority_arr)

        cur_x = to_tensor(np.array([self.memory[i]['state'] for i in indicies]), self.args.device)
        next_x = to_tensor(np.array([self.memory[i]['next_state'] for i in indicies]), self.args.device)
        action = to_tensor(np.array([self.memory[i]['action'] for i in indicies]), self.args.device)
        reward_vec = to_tensor(np.array([self.memory[i]['r'] for i in indicies]), self.args.device)
        terminal = to_tensor(np.array([self.memory[i]['terminal'] for i in indicies]), self.args.device)
        ## For back compat:
        if not 'oracle_action' in self.memory[0]:
            oracle_action = to_tensor(np.array([self.memory[i]['action'] for i in indicies]), self.args.device)
        else:
            oracle_action = to_tensor(np.array([self.memory[i]['oracle_action'] for i in indicies]), self.args.device)
        return cur_x, next_x, action,  reward_vec, terminal, oracle_action

    def learn(self):
        cur_x, next_x, action, reward_vec, terminal, oracle_action = self.sample()
        loss = self.get_loss(cur_x, next_x, action, reward_vec, terminal, oracle_action)
        return loss.mean()

    def search_envelope(self, x, preference, use_target_q, random_pref= None):
        B, _, Xd = x.size()
        if random_pref is None:
            random_pref = sample_preference_vector(B, True, self.args)
        exp_x = x.unsqueeze(0)
        exp_x = exp_x.expand([B, B, 1, Xd]).contiguous().view([B*B, 1, Xd])
        _, Rd, _ = random_pref.size()
        exp_pref = random_pref.unsqueeze(0)
        exp_pref   = random_pref.expand([B, B, Rd, 1]).transpose(0,1).contiguous().view([B*B, Rd, 1])
        # Get Q for expanded list of preferences
        if use_target_q:
            q_all_a = self.target_q(exp_x, exp_pref)
        else:
            q_all_a = self.q(exp_x, exp_pref)

        _, _, Ad = q_all_a.size()
        q_all_a = q_all_a.view([B, B, Ad, Rd])
        # Select best A given actual preferences
        q_all_a_pref_aligned = (q_all_a * preference.view([1,B, 1, Rd])).sum(dim=-1)
        q_all_a_pref_aligned = q_all_a_pref_aligned.view(B, B*Ad, 1)
        max_value_env, max_indx = torch.max(q_all_a_pref_aligned, dim=1)

        _max_indx = max_indx.unsqueeze(-1).expand([B, 1, Rd]).long()
        q_all_a_flat = q_all_a.view([B, B*Ad, Rd])
        estimated_q_env = (q_all_a_flat.gather(index=_max_indx, dim=1)).squeeze(1)

        max_value_env = max_value_env.squeeze(1)
        raw_rec_env = max_indx.squeeze(1) % Ad

        return max_value_env, raw_rec_env, estimated_q_env



    def forward(self, x, preference, batch):
        batch_size = x.size()[0]
        value  = (self.q(x.squeeze(1), preference) * preference).sum(dim=1)
        max_value, raw_rec = torch.max(value, dim=1)

        if self.args.envelope_inference:
            max_value_env, raw_rec_env, _ = self.search_envelope(x, preference, False)
            raw_rec =  (max_value > max_value_env).float() * raw_rec.float() +  (max_value <= max_value_env).float() * raw_rec_env.float()

        recommendation = (raw_rec + self.args.min_screening_interval_in_6mo_counts).float()

        if self.training:
            if hasattr(self.args, 'imitate_oracle') and  self.args.imitate_oracle:
                ## Return only min step if imitating oracle. This way all oracle transitions are sampled instead of only oracle trajectory
                y = batch['y']
                min_tensor = torch.ones_like(y).to(y.device) * self.args.min_screening_interval_in_6mo_counts
                is_pad = (y == PAD_Y_VALUE).float()
                return min_tensor if self.args.sample_all_oracle_transitions else (1-is_pad)*y + is_pad * min_tensor
            oracle_thresh = self.oracle_epsilon
            random_thresh = self.oracle_epsilon + self.args.epsilon
            do_oracle = (torch.Tensor(np.random.random(batch_size)) < oracle_thresh ).float().to(self.args.device)
            oracle_action = batch['y'].squeeze(1)
            do_oracle = (oracle_action != PAD_Y_VALUE).float() * do_oracle
            do_random = (torch.Tensor(np.random.random(batch_size)) < random_thresh).float().to(self.args.device) * (1-do_oracle)
            random_actions = np.random.choice(self.num_actions, batch_size) + self.args.min_screening_interval_in_6mo_counts
            random_actions = torch.Tensor(random_actions).to(self.args.device).float()
            recommendation = (recommendation * (1- do_random) + do_random * random_actions) * (1-do_oracle) + do_oracle * oracle_action
        return recommendation.float().unsqueeze(-1)



@RegisterModel("linear_policy")
class LinearPolicy(AbstractNeuralScreeningModel):
    def __init__(self, args):
        super(LinearPolicy, self).__init__(args)
        layers = [

                    nn.BatchNorm1d(self.input_dim),
                    nn.Linear(args.risk_dimension, self.num_actions * self.reward_dim)
                ]
        self.model = nn.Sequential(nn.Linear(args.risk_dimension, self.num_actions * self.reward_dim))
        self.target = copy.deepcopy(self.model)



@RegisterModel("mlp_policy")
class MLPPolicy(AbstractNeuralScreeningModel):
    def __init__(self, args):
        super(MLPPolicy, self).__init__(args)
        model_layers = []
        cur_dim = self.input_dim
        for layer in range(args.num_layers):
            bn = nn.BatchNorm1d(cur_dim)
            linear_layer = nn.Linear(cur_dim, args.hidden_dim)
            cur_dim = args.hidden_dim
            model_layers.extend( [bn, linear_layer, nn.ReLU()])

        bn_final = nn.BatchNorm1d(args.hidden_dim)
        fc_final =nn.Linear(args.hidden_dim, self.num_actions * self.reward_dim)
        model_layers.extend([bn_final, fc_final])

        self.model = nn.Sequential(*model_layers)
        self.target = copy.deepcopy(self.model)

