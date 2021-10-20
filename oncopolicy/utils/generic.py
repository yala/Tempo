import datetime
import hashlib
import numpy as np
from copy import deepcopy
import torch
import pdb

INVALID_DATE_STR = "Date string not valid! Received {}, and got exception {}"
ISO_FORMAT = '%Y-%m-%d %H:%M:%S'
CGMH_ISO_FORMAT ='%Y%m%d'
DAYS_IN_YEAR = 365
DAYS_IN_MO = 30
MAX_MO_TO_CANCER = 1200
MIN_MO_TO_CANCER = 3
MAX_PREFERNCES = 10.0
MIN_PREFERNCES = 0
EPSILON = 1e-3
AVG_MOMENTUM = 0.95
NUM_DIM_AUX_FEATURES = 7 ## Deprecated

class AverageMeter():
    def __init__(self):
        self.avg = 0
        self.first_update = True

    def reset(self):
        self.avg = 0
        self.first_update = True

    def update(self, val_tensor):
        val = val_tensor.item()
        if self.first_update:
            self.avg = val
            self.first_update = False
        else:
            self.avg = (AVG_MOMENTUM * self.avg) + (1-AVG_MOMENTUM) * val
        assert self.avg >= 0 and val >= 0

def get_aux_tensor(tensor, args):
    ## use of auxillary features for screen is deprecated
    return torch.zeros([tensor.size()[0], NUM_DIM_AUX_FEATURES]).to(tensor.device)


def to_numpy(tensor):
    return tensor.cpu().numpy()

def to_tensor(arr, device):
    return torch.Tensor(arr).to(device)

def sample_preference_vector(batch_size, sample_random, args):
    if sample_random:
        dist = torch.distributions.uniform.Uniform(MIN_PREFERNCES, MAX_PREFERNCES)
        preferences = dist.sample([batch_size, len(args.metrics), 1])
    else:
        preferences = torch.ones(batch_size, len(args.metrics), 1)

    preferences *= torch.tensor(args.fixed_preference).unsqueeze(0).unsqueeze(-1)

    preferences = preferences + EPSILON
    preferences = (preferences / (preferences).sum(dim=1).unsqueeze(-1))
    return preferences.to(args.device)

def normalize_dictionary(dictionary):
    '''
    Normalizes counts in dictionary
    :dictionary: a python dict where each value is a count
    :returns: a python dict where each value is normalized to sum to 1
    '''
    num_samples = sum([dictionary[l] for l in dictionary])
    for label in dictionary:
        dictionary[label] = dictionary[label]*1. / num_samples
    return dictionary


def parse_date(iso_string):
    '''
    Takes a string of format "YYYY-MM-DD HH:MM:SS" and
    returns a corresponding datetime.datetime obj
    throws an exception if this can't be done.
    '''
    try:
        return datetime.datetime.strptime(iso_string, ISO_FORMAT)
    except Exception as e:
        raise Exception(INVALID_DATE_STR.format(iso_string, e))

def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()

def pad_array_to_length(arr, pad_token, max_length):
    arr = arr[:max_length]
    return  np.array( arr + [pad_token]* (max_length - len(arr)))

def fast_forward_exam_by_one_time_step(curr_exam, NUM_DAYS_IN_TIME_STEP):
    exam = deepcopy(curr_exam)
    est_date_of_last_followup = curr_exam['date'] + datetime.timedelta(days=int(DAYS_IN_YEAR * curr_exam['years_to_last_followup']))
    est_date_of_cancer = curr_exam['date'] + datetime.timedelta(days=int(DAYS_IN_MO * curr_exam['months_to_cancer']))
    exam['date'] = curr_exam['date'] + datetime.timedelta(days=int(NUM_DAYS_IN_TIME_STEP))
    exam['years_to_last_followup'] = (est_date_of_last_followup - exam['date']).days / DAYS_IN_YEAR
    exam['months_to_cancer'] =  (est_date_of_cancer - exam['date']).days / DAYS_IN_MO
    exam['has_cancer'] = exam['months_to_cancer'] < MIN_MO_TO_CANCER
    exam['time_stamp'] = curr_exam['time_stamp'] + 1
    return exam
