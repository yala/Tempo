import torch
import torch.nn as nn
from oncopolicy.models.factory import RegisterModel
from oncopolicy.datasets.personalized_screening import PAD_Y_VALUE
import pdb

ANNUAL_GUIDELINE_IN_6MO_INTERVALS = 2
BIANNUAL_GUIDELINE_IN_6MO_INTERVALS = 4
SWITCHING_AGE = 55

class AbstractDeterministicGuideline(nn.Module):
    def __init__(self, args):
        super(AbstractDeterministicGuideline, self).__init__()
        self.args = args

    def save_transition(self, cur_x, next_x, action, reward_vec, preference, cur_is_censored, next_is_censored):
        pass

    def reset(self):
        pass

    def learn(self):
        return torch.zeros(1).to(self.args.device)

@RegisterModel("max_frequency_guideline")
class MaxFreqGuideline(AbstractDeterministicGuideline):
    def __init__(self, args):
        super(MaxFreqGuideline, self).__init__(args)

    def forward(self, x, preference, batch):
        y = batch['y']
        return torch.ones_like(y).to(y.device)



@RegisterModel("annual_guideline")
class AnnualGuideline(AbstractDeterministicGuideline):
    def __init__(self, args):
        super(AnnualGuideline, self).__init__(args)

    def forward(self, x, preference, batch):
        y = batch['y']
        return torch.ones_like(y).to(y.device) * ANNUAL_GUIDELINE_IN_6MO_INTERVALS


@RegisterModel("biannual_guideline")
class BiAnnualGuideline(AbstractDeterministicGuideline):
    def __init__(self, args):
        super(BiAnnualGuideline, self).__init__(args)

    def forward(self, x, preference, batch):
        y = batch['y']
        return torch.ones_like(y).to(y.device) * BIANNUAL_GUIDELINE_IN_6MO_INTERVALS



@RegisterModel("age_based_guideline")
class AgeBasedGuideline(AbstractDeterministicGuideline):
    def __init__(self, args):
        super(AgeBasedGuideline, self).__init__(args)

    def forward(self, x, preference, batch):
        y = batch['y']
        older = (batch['age'] >= 55).float().unsqueeze(1)
        annual = torch.ones_like(y).to(y.device) * ANNUAL_GUIDELINE_IN_6MO_INTERVALS
        biannual = torch.ones_like(y).to(y.device) * BIANNUAL_GUIDELINE_IN_6MO_INTERVALS
        recommendation = (older * biannual) + (1-older)*annual
        return recommendation

