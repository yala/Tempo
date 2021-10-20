import torch
from torch.utils import data
import numpy as np
from oncopolicy.datasets.factory import RegisterDataset
from oncopolicy.utils.generic import parse_date
from random import shuffle
import numpy as np
import tqdm
from collections import Counter
import warnings
warnings.simplefilter("ignore")

NUM_DAYS_IN_TIME_STEP = 182.5
MAX_STEPS = 30
MONTHS_IN_YEAR = 12
KI_POS_FRACTION = .019
SUMMARY_STR = "Contructed Risk Progression {} Dataset with {} samples from {} patients"

@RegisterDataset("risk_progression")
class Risk_Progression(data.Dataset):
    """
        A pytorch Dataset for the Risk Progression. i.e given observed risk scores at time
        stamps, predict missing time steps
    """

    def __init__(self, args, metadata, split_group):
        """Initializes the dataset.

        Constructs a standard pytorch Dataset object which
        can be fed into a DataLoader for batching.

        Arguments:
            args(object): Config.
            metadata(list): Whole risk and screen dataset
            split_group(str): The split group ['train'|'dev'|'test'].
        """

        super(Risk_Progression, self).__init__()
        self.args = args
        self.split_group = split_group
        self.dataset = []
        self.args.max_steps = MAX_STEPS
        self.max_steps = self.args.max_steps

        for patient_row in tqdm.tqdm(metadata):
            if patient_row['split_group'] != split_group:
                continue

            exams = patient_row['accessions']
            if len(exams) == 0:
                continue

            for exam in exams:
                exam['date'] = parse_date(exam['full_sdate'])

            exams = sorted(exams, key = lambda exam: exam['date'])
            min_date = exams[0]['date']

            ## Get time steps
            time_stamps = (lambda exams, min_date : [ int((e['date'] - min_date).days // NUM_DAYS_IN_TIME_STEP) for e in exams])(exams, min_date)
            ## Create vector with pads
            x = np.zeros([self.max_steps, args.risk_dimension])
            observed = np.zeros(self.max_steps)
            min_month_to_cancer = min([exam['months_to_cancer'] for exam in exams])
            ever_has_cancer = min_month_to_cancer < (args.risk_dimension * MONTHS_IN_YEAR)

            progression_censor_time = 0
            for i,t in enumerate(time_stamps):
                probs = exams[i]['probs']
                if args.recallibrate:
                    probs = recallibrate(probs, args)
                x[t,:] = probs
                observed[t] = 1
                progression_censor_time = t

            pos_neg_step_list = [(t, t+1) for t in time_stamps if not t+1 in time_stamps and t+1 < MAX_STEPS ]
            neg_pos_step_list = [(t-1, t) for t in time_stamps if not t-1 in time_stamps and t-1 >= 0]
            neg_neg_step_list = [(t, t+1) for t in range(MAX_STEPS) \
                if not t in time_stamps and not t+1 in time_stamps and t+1 < MAX_STEPS]
            pos_pos_step_list = [(t, t+1) for t in time_stamps if t+1 in time_stamps]

            pos_neg_step = [-1,-1] if len(pos_neg_step_list) == 0 else pos_neg_step_list[np.random.choice(len(pos_neg_step_list))]
            neg_pos_step = [-1,-1] if len(neg_pos_step_list) == 0 else neg_pos_step_list[np.random.choice(len(neg_pos_step_list))]
            neg_neg_step = [-1,-1] if len(neg_neg_step_list) == 0 else neg_neg_step_list[np.random.choice(len(neg_neg_step_list))]
            pos_pos_step = [-1,-1] if len(pos_pos_step_list) == 0 else pos_pos_step_list[np.random.choice(len(pos_pos_step_list))]

            if not 'age' in exam or exam['age'] == '':
                continue

            sample = {
                'x': x,
                'observed': observed,
                'progression_censor_time': progression_censor_time,
                'ssn': patient_row['ssn'],
                'exam':  exams[0]['accession'],
                'ever_has_cancer': ever_has_cancer,
                'age':int(exam['age']),
                'min_month_to_cancer': min_month_to_cancer,
                'obs_pos_neg_step': torch.LongTensor(pos_neg_step),
                'obs_neg_pos_step': torch.LongTensor(neg_pos_step),
                'obs_neg_neg_step': torch.LongTensor(neg_neg_step),
                'obs_pos_pos_step': torch.LongTensor(pos_pos_step)
            }

            ## Add to dataset
            self.dataset.append(sample)

        if len(self.dataset) > 0:
            if 'ki' in self.args.metadata_pickle_path and self.args.task == 'progression' and split_group != 'train':
                print("Re sampling data to have {} cancer rate".format(KI_POS_FRACTION))
                pos_data = [d for d in self.dataset if d['ever_has_cancer']]
                neg_data = [d for d in self.dataset if not d['ever_has_cancer']]
                target_neg_size = int((float(len(pos_data)) / KI_POS_FRACTION) - len(pos_data))
                sampled_dataset = np.random.choice(neg_data, target_neg_size).tolist()
                sampled_dataset.extend(pos_data)
                self.dataset = sampled_dataset

            label_dist = [d['ever_has_cancer'] for d in self.dataset]
            label_counts = Counter(label_dist)
            weight_per_label = 1./ len(label_counts)
            label_weights = {
                label: weight_per_label/count for label, count in label_counts.items()
            }
            if args.class_bal:
                print("Label weights are {}".format(label_weights))
            self.weights = [ label_weights[d['ever_has_cancer']] for d in self.dataset]

            print(SUMMARY_STR.format(split_group.upper(), len(self.dataset), len(set([d['ssn'] for d in self.dataset])) ))

    @staticmethod
    def set_args(args):
        args.task = 'progression'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def recallibrate(prob_arr, args):
    name = 'mgh'
    if 'ki' in args.metadata_pickle_path:
        name = 'ki'
    elif 'cgmh' in args.metadata_pickle_path:
        name = 'cgmh'
    elif 'emory' in args.metadata_pickle_path:
        name = 'emory'
    else:
        assert 'mgh' in args.metadata_pickle_path

    if name == 'mgh':
        return prob_arr

    name_to_stats = {
        'mgh': { 'mean':[0.15359685, 0.19702862, 0.19702862, 0.23001482, 0.23001482]  , 'std': [0.17467996, 0.16115864, 0.16115864, 0.1486038, 0.1486038]},
        'ki': { 'mean':[0.1037198, 0.14167216, 0.14167216, 0.17635849, 0.17635849]  , 'std': [ 0.16004314, 0.1529322, 0.1529322, 0.14634906, 0.14634906]},
        'cgmh': { 'mean':[0.11937303, 0.15833929, 0.15833929, 0.19670045, 0.19670045]  , 'std': [0.14482108, 0.13502893, 0.13502893, 0.12590846, 0.12590846]},
        'emory': { 'mean':[0.21119267, 0.25419354, 0.25419354, 0.28252363, 0.28252363]  , 'std': [0.20999531, 0.18940358, 0.18940358, 0.17317928, 0.17317928]},
    }

    for i in range(len(prob_arr)):
        prob_arr[i] = (prob_arr[i] - name_to_stats[name]['mean'][i]) / name_to_stats[name]['std'][i]
        prob_arr[i] = (prob_arr[i] * name_to_stats['mgh']['std'][i]) + name_to_stats['mgh']['mean'][i]

    return prob_arr
