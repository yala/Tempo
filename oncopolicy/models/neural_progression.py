import torch
import torch.nn as nn
from oncopolicy.models.factory import RegisterModel
from oncopolicy.models.cumulative_probability_layer import Cumulative_Probability_Layer
import pdb


class AbstractNeuralProgressionModel(nn.Module):
    '''
        Learned risk progression model. Predict
        observed risk at next timestep from last observation (real or predicted)
    '''
    def __init__(self, args):
        super(AbstractNeuralProgressionModel, self).__init__()
        self.args = args
        self.max_steps = args.max_steps
        self.input_dim = self.args.risk_dimension
        if hasattr(self.args, 'num_rnn_layers'): # For backward compat
            self.args.num_layers = self.args.num_rnn_layers


    def init_history(self, x):
        B = x.size()[0]
        return torch.zeros([self.args.num_layers, B, self.args.hidden_dim]).to(x.device)

    def pred_one_step(self, inp, hist):
        pass

    def get_logprob(self, z):
        z = z.unsqueeze(-1)
        return torch.log_softmax(torch.cat([z*0, z], dim=-1), dim=-1)

    def get_prob(self, z):
        '''
            Get probs from raw logits.
            first index of z is always input and is already a probablity
        '''
        mask = torch.zeros_like(z)
        mask[:,0,:] = 1
        return z*mask + torch.sigmoid(z)*(1-mask)

    def forward(self, x, batch):
        '''
            Forward func used in training/eval risk progression model.
            args:
            - x: tensor of shape [B, self.max_steps, args.risk_dimension], with 0s for unobserved
            - batch: full batch obj, contains 'oberved tensor'
            returns:
            - z: tensor of shape [B, self.max_steps, args.risk_dimension], with last observed risk for each dim
        '''
        B, L, D = x.size()
        x = x.float()
        obsereved_key = 'observed' if 'observed' in batch else 'progression_observed'
        obs = batch[obsereved_key].float() # shape [B, self.max_steps]
        inp = x[:,0,:]
        z = [inp.unsqueeze(1)]
        hist = self.init_history(inp)
        hidden_arr = [hist.unsqueeze(-1)]
        for step in range(1,L):
            pred, hist = self.pred_one_step(inp, hist)
            hidden_arr.append(hist.unsqueeze(-1))
            z.append(pred.unsqueeze(1))
            step_observed = obs[:,step].unsqueeze(-1)
            if  not self.args.teacher_forcing_for_progression and self.training:
                inp = pred
            else:
                inp = (step_observed) * x[:,step,:] + (1 - step_observed)* pred

        z = torch.cat(z, dim=1)
        hidden = torch.cat(hidden_arr, dim=-1).detach()
        hidden = hidden.permute([1,3,0,2])
        hidden = hidden.reshape( [*hidden.size()[:2], -1]).contiguous()
        return z, hidden


@RegisterModel("linear")
class Linear(AbstractNeuralProgressionModel):
    def __init__(self, args):
        super(Linear, self).__init__(args)
        self.fc = nn.Linear(self.input_dim, args.risk_dimension)

    def pred_one_step(self, inp, hist):
        return self.fc(inp), hist

@RegisterModel("linear_w_cum_hazard")
class Linear_Cum_Hazard(AbstractNeuralProgressionModel):
    def __init__(self, args):
        super(Linear_Cum_Hazard, self).__init__(args)
        self.cum_hazard = Cumulative_Probability_Layer(self.input_dim, args, args.risk_dimension)

    def pred_one_step(self, inp, hist):
        return self.cum_hazard(inp), hist

@RegisterModel("mlp_w_cum_hazard")
class MLP_Cum_Hazard(AbstractNeuralProgressionModel):

    def __init__(self, args):
        super(MLP_Cum_Hazard, self).__init__(args)
        self.fc1 = nn.Linear(self.input_dim, args.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 =nn.Linear(args.hidden_dim, args.hidden_dim)
        self.cum_hazard = Cumulative_Probability_Layer(args.hidden_dim, args, args.risk_dimension)

    def pred_one_step(self, inp, hist):
        h = self.relu(self.fc1(inp))
        h = self.relu(self.fc2(h))
        return self.cum_hazard(h), hist


@RegisterModel("gru_w_cum_hazard")
class GRU_Cum_Hazard(AbstractNeuralProgressionModel):

    def __init__(self, args):
        super(GRU_Cum_Hazard, self).__init__(args)
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=args.hidden_dim, num_layers=args.num_layers,
            batch_first=True)
        self.cum_hazard = Cumulative_Probability_Layer(args.hidden_dim, args, args.risk_dimension)

    def pred_one_step(self, inp, hist):
        hidden, new_state = self.gru(inp.unsqueeze(1), hist)
        pred = self.cum_hazard(hidden.squeeze(1))
        return pred, new_state
