import torch
import torch.nn as nn
from oncopolicy.models.factory import RegisterModel

import pdb

class AbstractDeterministicGuideline(nn.Module):
    def __init__(self, args):
        super(AbstractDeterministicGuideline, self).__init__()
        self.args = args
        self.max_steps = args.max_steps

    def get_logprob(self, z):
        z = z.unsqueeze(1)
        return torch.log(torch.cat([1-z, z], dim =1))

    def get_prob(self, z):
        return z


@RegisterModel("last_observed_risk")
class LastObservedRisk(AbstractDeterministicGuideline):
    '''
        Deterministic risk progression model. Predict
        observed risk doesnt change from last observation
    '''
    def __init__(self, args):
        super(LastObservedRisk, self).__init__(args)
        self.max_pool = nn.MaxPool1d(kernel_size=self.max_steps, stride=1)

    def forward(self, x, batch):
        '''
            Forward func used in training/eval risk progression model.
            args:
            - x: tensor of shape [B, self.max_steps, args.risk_dimension], with 0s for unobserved
            - batch: full batch obj, contains 'oberved tensor'
            returns:
            - z: tensor of shape [B, self.max_steps, args.risk_dimension], with last observed risk for each dim
        '''
        B, _, D = x.size()
        obsereved_key = 'observed' if 'observed' in batch else 'progression_observed'
        obs = batch[obsereved_key] # shape [B, self.max_steps]

        indicies = torch.arange(start=0, end=self.max_steps).unsqueeze(0).expand([B,self.max_steps]).to(self.args.device)

        obs_indicies = (obs.float() * indicies.float()).unsqueeze(1)
        obs_indicies_w_pad = torch.cat([torch.zeros([B, 1, self.max_steps]).to(self.args.device), obs_indicies[:,:,:-1]], dim=-1)
        indices_of_most_recent = self.max_pool(obs_indicies_w_pad).long().transpose(1,2).expand(B, self.max_steps, D)

        z = torch.gather(x, dim=1, index=indices_of_most_recent)
        return z, None

@RegisterModel("static_risk")
class StaticRisk(AbstractDeterministicGuideline):
    '''
        Deterministic risk progression model. Predict
        observed risk doesnt change from first observation. Assume static
    '''
    def __init__(self, args):
        super(StaticRisk, self).__init__(args)

    def forward(self, x, batch):
        '''
            Forward func used in training/eval risk progression model.
            args:
            - x: tensor of shape [B, self.max_steps, args.risk_dimension], with 0s for unobserved
            - batch: full batch obj, contains 'oberved tensor'
            returns:
            - z: tensor of shape [B, self.max_steps, args.risk_dimension], with last observed risk for each dim
        '''
        z = x[:,0,:].unsqueeze(1).expand_as(x).contiguous()
        return z, None


@RegisterModel("random")
class Random(AbstractDeterministicGuideline):
    '''
        Predict rand risk at each timestep.
    '''
    def __init__(self, args):
        super(Random, self).__init__(args)

    def forward(self, x, batch):
        '''
            Forward func used in training/eval risk progression model.
            args:
            - x: tensor of shape [B, self.max_steps, args.risk_dimension], with 0s for unobserved
            - batch: full batch obj, contains 'oberved tensor'
            returns:
            - z: tensor of shape [B, MAX_STEPS, args.risk_dimension], with last observed risk for each dim
        '''
        z = torch.sigmoid( torch.randn_like(x).to(x.device))
        return z, None

