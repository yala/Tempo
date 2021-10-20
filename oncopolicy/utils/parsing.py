import pickle
import numpy as np
import argparse
import torch
import os
import pwd
from oncopolicy.datasets.factory import get_dataset_class
from oncopolicy.utils.generic import AverageMeter
from oncopolicy.metrics.factory import get_metric_keys


BATCH_SIZE_SPLIT_ERR = 'batch_size (={}) should be a multiple of batch_splits (={})'
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
RACE_CODE_TO_NAME = {       1: 'White',
                            2: 'African American',
                            3: 'Other',
                            4: 'Asian or Pacific Islander',
                            5: 'Other',
                            6: 'Other',
                            7: 'Other',
                            8: 'Other',
                            9: 'Asian or Pacific Islander',
                            10: 'Asian or Pacific Islander',
                            11: 'Asian or Pacific Islander',
                            12: 'Asian or Pacific Islander',
                            13: 'Asian or Pacific Islander'
                            }

def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    '''
    jobs = [""]
    experiment_axies = []
    search_spaces = config['search_space']

    # Support a list of search spaces, convert to length one list for backward compatiblity
    if not isinstance(search_spaces, list):
        search_spaces = [search_spaces]


    for search_space in search_spaces:
        # Go through the tree of possible jobs and enumerate into a list of jobs
        for ind, flag in enumerate(search_space):
            possible_values = search_space[flag]
            if len(possible_values) > 1:
                experiment_axies.append(flag)

            children = []
            if len(possible_values) == 0 or type(possible_values) is not list:
                raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
            for value in possible_values:
                for parent_job in jobs:
                    if type(value) is bool:
                        if value:
                            new_job_str = "{} --{}".format(parent_job, flag)
                        else:
                            new_job_str = parent_job
                    elif type(value) is list:
                        val_list_str = " ".join([str(v) for v in value])
                        new_job_str = "{} --{} {}".format(parent_job, flag,
                                                          val_list_str)
                    else:
                        new_job_str = "{} --{} {}".format(parent_job, flag, value)
                    children.append(new_job_str)
            jobs = children

    return jobs, experiment_axies

def parse_args():
    parser = argparse.ArgumentParser(description='OncoPolicy Classifier')
    # setup
    parser.add_argument('--run_prefix', default="snapshot", help="what to name this type of model run")
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--do_subgroup_eval',  action='store_true', default=False, help="Rerun test on diff subgroups")
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # data
    parser.add_argument('--task', type=str, default='screening', help="Type of task. fit a screening policy or learn a risk progression model")
    parser.add_argument('--risk_dimension', type=int, default=5, help='Max followup risk is defined over. 5 for 5 years')
    parser.add_argument('--recallibrate', action='store_true', default=False, help="Recallibrate mirai probs per test set.")
    parser.add_argument('--max_screening_interval_in_6mo_counts', type=int, default=6, help='Max half-years before next screening. 6 for 3 years, 4 for 2 years, etc.')
    parser.add_argument('--min_screening_interval_in_6mo_counts', type=int, default=1, help='Min half-years before next screening. 1 for 6 months, etc.')
    parser.add_argument('--dataset', default='mnist', help='Name of dataset from dataset factory to use [default: mnist]')
    parser.add_argument('--use_all_trajec_for_eval', action='store_true', default=False, help='Whether or not to use all starting points in dev set')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 8]')
    parser.add_argument('--metadata_pickle_path', type=str, default='raw_data/ki_trajectories/mirai_trajectories.p.with_splits', help='path of metadata pickle file.')
    parser.add_argument('--metadata_csv_path', type=str, default='raw_data/cgmh_trajectories/cancer_dates.csv', help='path of auxillary information of trajectories.')
    parser.add_argument('--subgroup_metadata_path', type=str, default='/data/rsg/mammogram/pgmikhael/MGH_ACC_TO_X.pkl', help='path of metadata pickle file.')
    parser.add_argument('--get_conf_intervals',  action='store_true', default=False, help="Use conf intervals in reporting all metrics")
    parser.add_argument('--use_callibrator',  action='store_true', default=False, help="Use callibrator before using MSE metrics")
    parser.add_argument('--callibrator_path', type=str, default='raw_data/MIRAI_FULL_PRED_RF.callibrator.p', help='where to load the callibrator')
    # sampling
    parser.add_argument('--class_bal', action='store_true', default=False, help='Wether to apply a weighted sampler to balance between the classes on each batch.')

    # regularization
    parser.add_argument('--envelope_margin_loss_lambda',  type=float, default=0.5,  help='lambda to weigh the Envelope margin loss.')
    parser.add_argument('--envelope_margin',  type=float, default=0.5,  help='Size of the envelope margin loss.')
    parser.add_argument('--envelope_inference', action='store_true', default=False, help="Search envelope of preferences for inference")
    parser.add_argument('--imitate_oracle', action='store_true', default=False, help="Switch model to do imitation learning only.")

    parser.add_argument('--sample_all_oracle_transitions', action='store_true', default=False, help="Sample all transitions instead of oracle transitions only.")
    parser.add_argument('--homotopy_decay_rate',  type=float, default=0.99,  help='How to scale homtopy lambda every reset.')


    # learning
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='initial learning rate [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')
    parser.add_argument('--turn_off_model_reset', action='store_true', default=False, help="Don't reload the model to last best when reducing learning rate")
    parser.add_argument('--sample_random_preferences', action='store_true', default=False, help="Sample preferences at random with unit mean and a truncated guassian")
    parser.add_argument('--fixed_preference', nargs='*', default=[1.0, 3.0], help='List of preference weights')
    parser.add_argument('--num_estimates_for_dev', type=int, default=10, help='Number of loops through dev set to estimate randomized reward')


    parser.add_argument('--tuning_metric', type=str, default='loss', help='Metric to judge dev set results. Possible options include auc, loss, accuracy [default: loss]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--max_batches_per_train_epoch', type=int, default=10000, help='max batches to per train epoch. [default: 10000]')
    parser.add_argument('--max_batches_per_dev_epoch', type=int, default=10000, help='max batches to per dev epoch. [default: 10000]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 128]')
    parser.add_argument('--batch_splits', type=int, default=1, help='Splits batch size into smaller batches in order to fit gpu memmory limits. Optimizer step is run only after one batch size is over. Note: batch_size/batch_splits should be int [default: 1]')
    parser.add_argument('--dropout', type=float, default=0.25, help='Amount of dropout to apply on last hidden layer [default: 0.25]')
    parser.add_argument('--replay_size', type=int, default=10000, help='Amount transitions to store in experience replay size')
    parser.add_argument('--reward_decay_lambda', type=float, default=.99, help='How to weight rewards in one transition from here')
    parser.add_argument('--epsilon', type=float, default=.1, help='How often to take random actions to explore')
    parser.add_argument('--max_oracle_prob', type=float, default=0.50, help='Max how often to take oracle action to explore with oracle')
    parser.add_argument('--min_oracle_prob', type=float, default=0.10, help='Min how often to take oracle action to explore with oracle')
    parser.add_argument('--oracle_decay_rate', type=float, default=0.50, help='How much to decay oracle prob on every reset')
    parser.add_argument('--reset_rate', type=int, default=500, help='How many steps before reset target model, decay epsilon etc.')
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to dump the model')
    parser.add_argument('--results_path', type=str, default='logs/snapshot', help='where to save the result logs')
    parser.add_argument('--no_tuning_on_dev', action='store_true', default=False,  help='Train without tuning on dev (no adaptive lr reduction or saving best model based on dev)')
    parser.add_argument('--lr_reduction_interval', type=int, default=1, help='Number of epochs to wait before reducing lr when training without adaptive lr reduction.')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use, i.e 1.0 for all and 0 for none. Used for learning curve analysis.')

    # model
    parser.add_argument('--screening_model_name', type=str, default='annual_guideline', help="Form of screening policy model, i.e annual_guideline, some nn, etc.")
    parser.add_argument('--screening_snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--progression_model_name', type=str, default='last_observed_risk', help="Form of progression model, i.e last_observed_risk, some nn, etc, etc.")
    parser.add_argument('--progression_snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--hidden_dim', type=int, default=50, help='Hidden dim of linear layers in progression or screening model')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers used neural models.')
    parser.add_argument('--teacher_forcing_for_progression', action='store_true', default=False, help='Use teacher forcing when training risk progression model')
    parser.add_argument('--max_early_detection_benefit', type=int, default=18, help='Max number of months that yields an early detection benefit.')
    parser.add_argument('--use_pessimistic_detection_definition', action='store_true', default=False, help='Consider only screens in max window to offer benefit')

    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use in data_parallel.')
    parser.add_argument('--data_parallel', action='store_true', default=False, help='spread batch size across all available gpus. Set to false when using model parallelism. The combo of model and data parallelism may result in unexpected behavior')

    args = parser.parse_args()
    # Set args particular to dataset
    get_dataset_class(args).set_args(args)

    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.unix_username = pwd.getpwuid( os.getuid() )[0]
    args.metrics = sorted(get_metric_keys(screen_only = True))

    args.fixed_preference = [float(pref) for pref in args.fixed_preference ]
    args.step_index = 1
    assert len(args.fixed_preference) == len(args.metrics)


    # learning initial state
    args.optimizer_state = None
    args.current_epoch = None
    args.lr = None
    args.epoch_stats = None
    args.step_indx = 1

    if args.use_callibrator:
        args.callibrator = pickle.load(open(args.callibrator_path,'rb'))

    # Check whether certain args or arg combinations are valid
    validate_args(args)

    np.random.seed(args.seed)
    args.average_meter_dict = {'loss': AverageMeter()}

    return args

def load_subgroups(args):
    metadata = pickle.load(open(args.subgroup_metadata_path,'rb'))

    subgroup_lambda = {'african american': lambda sample: 'African American' in metadata['acc_to_race'][sample['exam']],
                   'asian': lambda sample: 'Asian' in metadata['acc_to_race'][sample['exam']] ,
                    'white': lambda sample: 'White' in metadata['acc_to_race'][sample['exam']] ,
                   '<= 55': lambda sample: metadata['acc_to_age'][sample['exam']] <= 55,
                   '> 55': lambda sample: metadata['acc_to_age'][sample['exam']] > 55,
                   'Non-dense': lambda sample: metadata['acc_to_density'][sample['exam']] in [1,2],
                   'Dense': lambda sample: metadata['acc_to_density'][sample['exam']] in [3,4],
                    }
    return subgroup_lambda


def validate_args(args):
    """Checks whether certain args or arg combinations are valid.

    Raises:
        Exception if an arg or arg combination is not valid.
    """

    if args.batch_size % args.batch_splits != 0:
        raise ValueError(BATCH_SIZE_SPLIT_ERR.format(args.batch_size, args.batch_splits))

    assert args.task in ['screening', 'progression']
