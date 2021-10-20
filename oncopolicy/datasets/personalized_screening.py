import torch
from torch.utils import data
import numpy as np
import csv
from oncopolicy.datasets.factory import RegisterDataset
from oncopolicy.datasets.risk_progression import Risk_Progression, MAX_STEPS, recallibrate
from oncopolicy.utils.generic import pad_array_to_length, fast_forward_exam_by_one_time_step, parse_date, CGMH_ISO_FORMAT
from random import shuffle
import datetime
import tqdm
import numpy as np
import warnings
from collections import Counter
warnings.simplefilter("ignore")

NUM_DAYS_IN_TIME_STEP = 182.5
NUM_STEP_IN_YEAR = 2
MIN_MO_TO_CANCER = 3
MO_IN_STEP = 6
MONTHS_IN_YEAR = 12
NO_CANCER_TIME = 120
NUM_DAYS_IN_MONTH = 30
PAD_Y_VALUE = -1.0
KI_POS_FRACTION = .019
SUMMARY_STR = "Contructed Personalized Screening {} Dataset with {} samples from {} exams from {} patients. {} eventually have cancer"
CANCER_STEP_TOO_HIGH_WARNING = "WARNING: Time step at cancer for {} is {}. This is > MAX_STEPS(={}). Unexpected behavior, lowering to MAX_STEPS"

@RegisterDataset("personalized_screening")
class Personalized_Screening(data.Dataset):
    """
        A pytorch Dataset for the Personalized Screening. i.e given observed risk scores at time,
        predict optimal time for next screening.
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

        super(Personalized_Screening, self).__init__()
        self.args = args
        self.split_group = split_group
        progression = Risk_Progression(args, metadata, split_group).dataset
        self.ssn_to_progression = {prog['ssn']:prog for prog in progression}
        self.max_steps = self.args.max_steps
        self.use_external_dates = False
        if 'cgmh' in self.args.metadata_pickle_path:
            self.use_external_dates = True
            screening_dates = [r for r in csv.DictReader(open(self.args.metadata_csv_path,'r'))]
            self.screening_dates = {r['pseudo ID']: r for r in screening_dates}

        self.dataset = []
        all_mo_to_cancer = []
        for patient_row in tqdm.tqdm(metadata):
            if patient_row['split_group'] != split_group or not patient_row['ssn'] in self.ssn_to_progression:
                continue

            ssn, exams = patient_row['ssn'], patient_row['accessions']
            progression_info = self.ssn_to_progression[ssn]
            progression_x, progression_observed = progression_info['x'], progression_info['observed']
            progression_censor_time = progression_info['progression_censor_time']

            for exam in exams:
                exam['date'] = parse_date(exam['full_sdate'])
                exam['has_cancer'] = exam['months_to_cancer'] < MIN_MO_TO_CANCER
                all_mo_to_cancer.append(exam['months_to_cancer'] // 6)
            exams = sorted(exams, key = lambda exam: exam['date'])
            min_date = exams[0]['date']

            time_stamps = (lambda exams, min_date : [ int((e['date'] - min_date).days // NUM_DAYS_IN_TIME_STEP) for e in exams])(exams, min_date)
            for time_stamp, exam in zip(time_stamps, exams):
                exam['time_stamp'] = time_stamp
            ever_has_cancer = exams[0]['months_to_cancer'] < (min(NO_CANCER_TIME, self.max_steps * MO_IN_STEP))
            time_step_at_cancer, ever_has_cancer, raw_time_step_at_cancer = self.get_time_step_at_cancer(exams, ever_has_cancer, ssn)
            exams_pre_cancer = [e for e in exams if e['time_stamp'] < time_step_at_cancer]
            if len(exams_pre_cancer) == 0:
                continue
            last_neg_time_step = self.get_last_neg_time_step(exams, raw_time_step_at_cancer, ssn)

            # Last roll out step must start before this time stamp
            rollout_censor_time_stamp = time_step_at_cancer if ever_has_cancer else last_neg_time_step
            for i, exam in enumerate(exams):
                if exam['has_cancer']:
                    break

                if not args.use_all_trajec_for_eval:
                    # if split group is dev or test, only use first exam for rollout:
                    if i > 0 and split_group != "train":
                        break

                followup_exams = [e for e in exams[i+1:] if e['time_stamp'] <= time_step_at_cancer]

                num_screens_historical = 1 + (lambda followup_exams, rollout_censor_time_stamp: len([followup for followup in followup_exams if followup['time_stamp'] <= rollout_censor_time_stamp]))(followup_exams, rollout_censor_time_stamp)
                if not self.is_valid_sample(exam) or exam['time_stamp'] > rollout_censor_time_stamp or not 'age' in exam or exam['age'] == '':
                    continue

                x = exam['probs']
                if args.recallibrate:
                    x = recallibrate(x, args)
                y_seq, y_hist, y_observed = self.get_y_seq(exam, followup_exams, time_step_at_cancer, last_neg_time_step, ever_has_cancer)
                y_seq, y_hist, y_observed = pad_array_to_length(y_seq, PAD_Y_VALUE, self.max_steps), pad_array_to_length(y_hist, PAD_Y_VALUE, self.max_steps),  pad_array_to_length(y_observed, 0, self.max_steps)
                sample = {
                        # identifiers
                        'ssn': ssn,
                        'exam':exam['accession'] if not 'emory' in args.metadata_pickle_path else '{}\t{}'.format(ssn,exam['accession']),
                        # risk information
                        'x':x,
                        'time_stamp': exam['time_stamp'],
                        'risk_progression_x': progression_x,
                        'progression_observed': progression_observed,
                        'num_screens_historical': num_screens_historical,
                        'age':int(exam['age']),
                        #historial information
                        'y_historical': y_hist,
                        # oracle trajectory information
                        'y_seq': y_seq,
                        'y_observed': y_observed,
                        'ever_has_cancer': ever_has_cancer,
                        'last_neg_time_step': last_neg_time_step,
                        'time_step_at_cancer': time_step_at_cancer,
                        'rollout_censor_time_stamp': rollout_censor_time_stamp
                        }
                self.dataset.append(sample)

        if len(self.dataset) > 0:

            if 'ki' in self.args.metadata_pickle_path and split_group != 'train':
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
            print('all_mo_to_cancer Counter: {}'.format(Counter(all_mo_to_cancer)))
            fraction_pos = label_counts[True] / sum(list(label_counts.values()))
            print(SUMMARY_STR.format(split_group.upper(), len(self.dataset), len(set([d['ssn'] for d in self.dataset])), len(set([ d['ssn'] + d['exam'] for d in self.dataset])), fraction_pos))
            print("Pos patients", len( set(d['ssn'] for d in self.dataset if d['ever_has_cancer']))) 

            if not split_group == 'train':
                self.dataset = self.dataset * args.num_estimates_for_dev

    @staticmethod
    def set_args(args):
        args.task = 'screening'

    def is_valid_sample(self, exam, allow_current_cancer= False):
        '''
            Include exam if:
                Has enough followup to make a decision:
                    - last neg followup > max_screening_interval or
                    - future cancer happens
                and :
                    - Doesn't have cancer now:
                        -   defined by months to cancer within 6 months.
        '''
        not_cancer_now =  not exam['has_cancer']
        min_followup_years = self.args.max_screening_interval_in_6mo_counts // 2
        valid_neg = exam['years_to_last_followup'] >= min_followup_years
        valid_pos = exam['months_to_cancer'] < NO_CANCER_TIME

        return (not_cancer_now or allow_current_cancer) and (valid_neg or valid_pos)

    def get_y_seq(self, exam, followup_exams, time_step_at_cancer, last_neg_time_step, ever_has_cancer):
        '''
            Given an exam starting at time_step i, fast forward exam
            one time step at a time and compute oracle label until the trajectory is censored.

            args:
            - exam: observed exam starting at some current time step
            - followup_exams: observed exams happening after this exam

            returns:
            y_seq: oracle screening decisions from timestep (i) til censoring.
            y_hist: actual screening decisions (viewed as time to next screen) from timestep (i) til censoring.
            y_obs: equal length arr to y_seq indicating which value is a pad.
        '''
        curr_step = exam['time_stamp']
        curr_exam = exam
        curr_followup_exams = followup_exams
        curr_followup_exams = sorted(curr_followup_exams, key = lambda exam: exam['date'])
        y_arr = [PAD_Y_VALUE] * curr_step
        y_hist = [PAD_Y_VALUE] * curr_step
        y_obs = [0] * curr_step

        while curr_step < self.max_steps and curr_step < time_step_at_cancer and self.is_valid_sample(curr_exam, allow_current_cancer=True):
            oracle_decison = get_oracle_prediction(curr_step, time_step_at_cancer, last_neg_time_step, ever_has_cancer, self.args)
            historical_decision = get_historical_decision(curr_step, time_step_at_cancer, last_neg_time_step, curr_followup_exams, self.args)
            y_arr.append(oracle_decison)
            y_hist.append(historical_decision)
            y_obs.append(1)
            curr_step += 1

            if len(curr_followup_exams) > 0 and curr_step == curr_followup_exams[0]['time_stamp']:
                curr_exam = curr_followup_exams[0]
            else:
                # update exam by 6 months into future, removing followups if necessary
                curr_exam = fast_forward_exam_by_one_time_step(curr_exam, NUM_DAYS_IN_TIME_STEP)
                assert curr_exam['time_stamp'] == curr_step

            curr_followup_exams = (lambda followup_exams, curr_exam: [followup for followup in followup_exams if followup['date'] > curr_exam['date']])(followup_exams, curr_exam)
            curr_followup_exams = sorted(curr_followup_exams, key = lambda exam: exam['date'])

        return y_arr, y_hist, y_obs

    def get_time_step_at_cancer(self, exams, ever_has_cancer, ssn):
        '''
            Get time stamp at time of cancer. If never has cancer, return MAX_STEPs, the universal censor time
        '''
        raw_time_step_at_cancer = exams[0]['time_stamp'] + (exams[0]['months_to_cancer'] // 6) if ever_has_cancer else NO_CANCER_TIME // 6
        time_step_at_cancer = min(raw_time_step_at_cancer, self.max_steps)
        if time_step_at_cancer > self.max_steps - 1:
            time_step_at_cancer = max(time_step_at_cancer, self.max_steps - 1)
            ever_has_cancer = False
        return time_step_at_cancer, ever_has_cancer, raw_time_step_at_cancer

    def get_last_neg_time_step(self, exams, time_step_at_cancer, ssn):
        '''
            Get time stamp of last exam not counted as cancer positive.
        '''
        if self.use_external_dates and time_step_at_cancer < 20:
            date_deltas = []
            cur_date = exams[0]['date']
            for i in range(1,12):
                date_str = self.screening_dates[ssn]['date{}'.format(i)].strip()
                if date_str == '':
                    continue
                try:
                    new_date = datetime.datetime.strptime(date_str, CGMH_ISO_FORMAT)
                except:
                    print("Date {} failed to convert".format(date_str))
                    continue
                if new_date > cur_date:
                    delta = (new_date - cur_date).days // NUM_DAYS_IN_TIME_STEP
                    if delta < time_step_at_cancer:
                        date_deltas.append(delta)
            if len(date_deltas) > 0:
                last_neg_time_step = max(date_deltas)
                assert last_neg_time_step >= 0
                assert last_neg_time_step < time_step_at_cancer
                return min(self.max_steps - 1, last_neg_time_step)

        last_neg_time_step_from_followups = exams[0]['time_stamp'] + exams[0]['years_to_last_followup'] * 2
        last_obs = max([e['time_stamp'] for e in exams if e['time_stamp'] < time_step_at_cancer])
        last_neg_time_step = max(last_neg_time_step_from_followups, last_obs)
        if last_neg_time_step >= time_step_at_cancer: ## Find last neg pre_cancer
            return max( [e['time_stamp'] for e in exams if e['time_stamp'] < time_step_at_cancer])
        return min(self.max_steps - 1, last_neg_time_step)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def get_historical_decision(curr_step, time_step_at_cancer, last_neg_time_step, followups, args):
    _followups = [f for f in followups if f['time_stamp'] > curr_step]
    if len(_followups) > 0:
        decision = min([f['time_stamp'] - curr_step for f in _followups])
    else:
        ## In case where no more know follows, fast fwd to next date of cancer or alst known neg
        next_known_time = min([time for time in [time_step_at_cancer, last_neg_time_step] if time > curr_step] )
        decision = next_known_time - curr_step
    return float(decision)


def get_oracle_prediction(cur_time_step, time_step_at_cancer, last_neg_time_step, ever_has_cancer, args):
    '''
        Given exam at cur_time_step, return optimal time to screen again.
        Note, this defines the oracle policy for this task.

        3 possible cases:
            1. No known cancer:
                y = max screening interval
            2. Pos before next neg exam (if any)
                y = min screening interval
            3. Pos after a neg exam
                y = min( max screening interval, time of last neg exam)

        returns:
            - oracle_decision: an float describing time to screen in half-years increments
    '''

    if ever_has_cancer:
        if cur_time_step >= last_neg_time_step:
            oracle_decision =  args.min_screening_interval_in_6mo_counts
        else:
            decision_lowerbound = max(last_neg_time_step - cur_time_step, args.min_screening_interval_in_6mo_counts)
            oracle_decision =  min(decision_lowerbound, args.max_screening_interval_in_6mo_counts)
    else:
        oracle_decision = args.max_screening_interval_in_6mo_counts
    return float(oracle_decision)

