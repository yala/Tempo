import numpy as np
import math
import sklearn.metrics
import torch
import torch.nn.functional as F
from oncopolicy.metrics.factory import compute_reward
from oncopolicy.utils.generic import sample_preference_vector, AverageMeter
from oncopolicy.utils.generic import get_aux_tensor

def progression_model_step(x, batch, models, optimizers, train_model, args):
    '''
    Single step of running progression model on the a batch x,y and computing
    the loss. Backward pass is computed if train_model=True and model has params.

    Returns various stats of this single forward and backward pass for this batch.


    args:
        - x: input features
        - batch: whole batch dict, can be used by various special args
        - models: dict of models. The main model, named "model" must return logit, hidden, activ
        - optimizers: dict of optimizers for models
        - train_model: whether or not to compute backward on loss
        - args: various runtime args such as batch_split etc
    returns:
        - loss: scalar for loss on batch as a tensor
        - additional_metrics: other metrics like regularization loss, etc.
        - preds: predicted labels as numpy array
        - ssns: patient ids for batch if available
    '''
    x = x.float()
    observed = batch['observed']
    # Ignore first obvs since copied over and not relevant for prediction
    observed[:,0] = 0
    raw_logits, _ = models['model'](x, batch)
    probs = models['model'].get_prob(raw_logits)
    indicies = torch.range(0,args.risk_dimension -1 ).view( [1,1,args.risk_dimension]).expand_as(x).to(x.device)
    known_x = torch.masked_select(x, observed.unsqueeze(-1).bool())
    raw_logits_for_known_x = torch.masked_select(raw_logits, observed.unsqueeze(-1).bool())
    probs_for_known_x = torch.masked_select(probs, observed.unsqueeze(-1).bool())
    known_index = torch.masked_select(indicies, observed.unsqueeze(-1).bool()).long()
    ## Get KL loss
    known_x_as_probs = torch.cat( [1 - known_x.unsqueeze(-1), known_x.unsqueeze(-1)], dim=1)
    logprobs = models['model'].get_logprob(raw_logits_for_known_x)
    loss = F.kl_div(logprobs, known_x_as_probs, reduction = 'batchmean') if observed.sum().item() > 0 else torch.zeros(1)
    loss /= args.batch_splits

    if train_model:
        args.average_meter_dict['loss'].update(loss)

    additional_metrics = {'main_loss': loss.item()}
    if args.use_callibrator and observed.sum().item() > 0 and 'mirai' in args.metadata_pickle_path.lower():
        calib_pred, calib_target, calib_index = probs_for_known_x.detach().cpu().numpy(), known_x.detach().cpu().numpy(), known_index.detach().cpu().numpy()
        for i in args.callibrator.keys():
            calib_pred[calib_index == i] =  args.callibrator[i].predict_proba(calib_pred[ calib_index == i].reshape(-1,1))[:,1]
            calib_target[calib_index == i] =  args.callibrator[i].predict_proba(calib_target[ calib_index == i].reshape(-1,1))[:,1]

        abs_five_years_deltas = calib_pred[ calib_index == args.risk_dimension - 1] - calib_target[ calib_index ==  args.risk_dimension - 1]
        avg_five_year_absolute_error = np.abs(abs_five_years_deltas).mean()
        additional_metrics['abs_five_year_error'] = avg_five_year_absolute_error

        calib_pred = torch.tensor(calib_pred).to(x.device)
        calib_target = torch.tensor(calib_target).to(x.device)
        additional_metrics['mse_loss'] = F.mse_loss(calib_pred, calib_target).item()

    # Optimizer not none if model is parameteric (i.e not a deterministic baseline)
    if train_model and optimizers['model'] is not None and loss.requires_grad:
        loss.backward()

    batch_probs = probs.data.cpu().numpy()
    batch_ssns = batch['ssn']
    batch_exams = batch['exam']
    return loss, additional_metrics, batch_probs, batch_ssns, batch_exams, [0]

def screening_model_step(x, batch, models, optimizers, train_model, args):
    '''
        Single step of running screening model on the a batch x,y and computing
        the loss. Backward pass is computed if train_model=True and model has params.

        Returns various stats of this single forward and backward pass for this batch.


        args:
        - x: input features
        - batch: whole batch dict, can be used by various special args
        - models: dict of models. The main model, named "model" must return logit, hidden, activ
        - optimizers: dict of optimizers for models
        - train_model: whether or not to compute backward on loss
        - args: various runtime args such as batch_split etc
        returns:
        - loss: scalar for loss on batch as a tensor
        - additional_metrics: other metrics like regularization loss, total_im from rollout etc.
        - preds: predicted labels as numpy array
        - golds: labels, numpy array version of arg y
        - ssns: patient ids for batch if available
        - exams: accessions for element in batch
    '''
    x = x.float()
    loss = 0
    total_obs_for_loss = 0
    batch_exams = batch['exam']
    batch_size = x.size()[0]
    batch_ssns = batch['ssn']
    y_seq = batch['y_seq'].float()
    y_historical = batch['y_historical'].float()
    y_observed = batch['y_observed']
    progression_observed = batch['progression_observed'].float()
    rollout_censor_time_stamp = batch['rollout_censor_time_stamp']
    will_get_cancer = batch['ever_has_cancer']
    progression_x = batch['risk_progression_x'].float()
    current_time_steps = batch['time_stamp']

    preference_vec = sample_preference_vector(batch_size, args.sample_random_preferences, args)

    screen_progression = [ current_time_steps.data.cpu().numpy().tolist() ]

    estimated_full_x = get_estimated_x(progression_x, progression_observed, models, batch, args).detach()
    aux_features = get_aux_tensor(batch['age'], args)
    first_step = True

    ### Environment rollout steps
    cur_is_censored = (current_time_steps >= rollout_censor_time_stamp)
    counter = 0
    while continue_rollout(cur_is_censored):
        counter += 1
        cur_x, y, y_hist, observed = get_vars_for_rollout_step(estimated_full_x, y_seq, y_historical, y_observed, aux_features, current_time_steps)
        batch['y'] = y # record this in batch dict for use by oracle
        batch['y_hist'] = y_hist
        action = models['model'](cur_x, preference_vec, batch)
        current_time_steps, next_is_censored = update_timestep(current_time_steps, action, rollout_censor_time_stamp, args.max_steps, args)
        screen_progression.append( current_time_steps.data.cpu().numpy().tolist() )
        screen_transition =  np.array(screen_progression).transpose()[:,-2:].tolist()
        transition_reward_vec = compute_reward(screen_transition, batch, args)

        next_x, _, _, _ = get_vars_for_rollout_step(estimated_full_x, y_seq, y_historical, y_observed, aux_features, current_time_steps)
        if train_model:
            models['model'].save_transition(cur_x, next_x, action, transition_reward_vec.t(), preference_vec, cur_is_censored, next_is_censored, y)
        cur_is_censored = next_is_censored

        if counter > 20:
            raise Exception("Roll out stuck in infinite loop.")

    screen_progression = np.array(screen_progression).transpose().tolist()
    trajectory_reward_vec = compute_reward(screen_progression, batch, args)
    total_reward = (trajectory_reward_vec.t().unsqueeze(-1) * preference_vec).squeeze(1).sum(dim=1).squeeze(1).cpu().numpy().tolist()
    named_metrics = {metric_name: trajectory_reward_vec[i].cpu().numpy().tolist() for i, metric_name in enumerate(args.metrics)}

    ## For cancer metrics, change denom to only cancers
    num_cancers = will_get_cancer.cpu().numpy().tolist()
    named_metrics['total_reward'] = total_reward

    ### model update
    loss = models['model'].learn()
    # Optimizer not none if model is parameteric (i.e not a deterministic baseline)
    if train_model and optimizers['model'] is not None and loss.requires_grad:
        loss.backward()

    return loss, named_metrics, screen_progression, batch_ssns, batch_exams, num_cancers

def get_estimated_x(progression_x, progression_observed, models, batch, args):
    '''
        Get estimated x from progression model if x is not observed
    '''
    progression_model = models['progression']
    logit, hidden = progression_model(progression_x, batch)
    estimated_x = progression_model.get_prob(logit)
    x = progression_x * (progression_observed.unsqueeze(-1)) + estimated_x * (1 - progression_observed.unsqueeze(-1))
    return x

def continue_rollout(censors):
    '''
        Given vector of timestamps, should we continue rolling out the batch.
        Check if any timestamp is not censored for lack of followup.

        assumes that time_stamps all bellow MAX_STEP and valid

        args:
        - time_stamp_tensor: stamp of current step of rollout for each element in batch dim [B]. Must be below max steps
        - rollout_censor_time_stamp: time_step where trajectories are censors. Either because has cancer at that step or last neg screen.

        return (bool) whether to continue rollout
    '''
    return not censors.bool().all().item()

def get_vars_for_rollout_step(x_seq, y_seq, y_hist, y_obs, aux_features, current_time_steps):
    '''
        Get relevant x, y, and observed info for curr rollout step.
    '''
    B, L, D = x_seq.size()
    x = torch.gather(x_seq, dim=1, index=current_time_steps.unsqueeze(-1).unsqueeze(-1).expand([B,1,D])).detach()
    y = torch.gather(y_seq, dim=1, index=current_time_steps.unsqueeze(-1))
    y_h = torch.gather(y_hist, dim=1, index=current_time_steps.unsqueeze(-1))
    obs = torch.gather(y_obs, dim=1, index=current_time_steps.unsqueeze(-1))

    aux_vec = aux_features.unsqueeze(1)
    x_w_aux = torch.cat([x, aux_vec], dim=-1)
    return x_w_aux, y, y_h, obs

def update_timestep(current_time_steps, pred, rollout_censor_time_stamp, max_steps, args):
    '''
        Given current time steps and recommendations of current model.
        update the time steps.

        returns time_steps
    '''
    B = current_time_steps.size()[0]
    can_do_step = (current_time_steps < rollout_censor_time_stamp).long()
    pred = torch.round(pred).long().squeeze(1)
    time_steps = (pred + current_time_steps) * can_do_step + (1 - can_do_step) * current_time_steps
    max_time_steps = (torch.ones_like(time_steps) * (max_steps - 1)).to(time_steps.device)
    time_steps = torch.min(time_steps, max_time_steps)
    cur_is_terminal = 1 - can_do_step
    return time_steps.detach(), cur_is_terminal

