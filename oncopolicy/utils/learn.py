import pickle
import json
import warnings
import torch
import numpy as np
from torch.utils import data
import sklearn.metrics
from collections import defaultdict, OrderedDict, Counter
from oncopolicy.metrics.factory import get_metrics_with_cis


def init_metrics_dictionary():
    '''
    Return empty metrics dict
    '''
    stats_dict = defaultdict(list)
    stats_dict['best_epoch'] = 0
    return stats_dict

def get_train_and_dev_dataset_loaders(args, train_data, dev_data, batch_size):
    '''
        Given arg configuration, return appropriate torch.DataLoader
        for train_data and dev_data

        returns:
        train_data_loader: iterator that returns batches
        dev_data_loader: iterator that returns batches
    '''
    if args.class_bal:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)
        train_data_loader = torch.utils.data.DataLoader(
                train_data,
                num_workers=args.num_workers,
                sampler=sampler,
                pin_memory=True,
                batch_size=batch_size,
                collate_fn=ignore_None_collate)
    else:
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=ignore_None_collate,
            pin_memory=True,
            drop_last=True)

    dev_data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False)

    return train_data_loader, dev_data_loader

def collate_eval_metrics(args, loss, preds, ssns, exams, metrics, stats_dict, key_prefix):

    stats_dict['{}_loss'.format(key_prefix)].append(loss)
    stats_dict['preds'] = preds
    stats_dict['exams'] = exams
    stats_dict['ssns'] = ssns
    log_statement = '--\nLoss: {:.6f} '.format(loss)
    for key in metrics:
        if 'list' in key:
            stats_dict['{}_{}'.format(key_prefix, key)] = metrics[key]

    for key in ['total_reward'] + get_metrics_with_cis():
        if key in metrics:
            stat_name = "{}_{}".format(key_prefix, key)
            stats_dict[stat_name].append(metrics[key])
            log_statement += "--{} {:.6f} ".format(stat_name, metrics[key])
    if args.task == 'screening':
        actions = []
        for pred_arr in preds:
            action_arr =  (lambda pred_arr: [pred_arr[i+1] - pred_arr[i] for i in range(len(pred_arr) -1) if pred_arr[i+1] != pred_arr[i] ])(pred_arr)
            actions.append(action_arr)
        stats_dict['actions'] = actions
        all_actions = []
        for action_arr in actions:
            all_actions.extend(action_arr)
        histogram =  Counter(all_actions)
        stats_dict['action_histogram'] = histogram
        log_statement += '--action_historgram {}'.format(histogram)

        stats_dict["{}_efficiency".format(key_prefix)] = -stats_dict['{}_mo_to_cancer'.format(key_prefix)][-1] / stats_dict['{}_annualized_mammography_cost'.format(key_prefix)][-1]
        log_statement += '--efficiency {}'.format(stats_dict["{}_efficiency".format(key_prefix)])
        if args.get_conf_intervals:
            stats_dict["{}_efficiency_lower_95".format(key_prefix)] = -stats_dict['{}_mo_to_cancer_lower_95'.format(key_prefix)][-1] / stats_dict['{}_annualized_mammography_cost_lower_95'.format(key_prefix)][-1]
            stats_dict["{}_efficiency_upper_95".format(key_prefix)] = -stats_dict['{}_mo_to_cancer_upper_95'.format(key_prefix)][-1] / stats_dict['{}_annualized_mammography_cost_upper_95'.format(key_prefix)][-1]
            log_statement += ' ({} , {})'.format(stats_dict["{}_efficiency_lower_95".format(key_prefix)], stats_dict["{}_efficiency_upper_95".format(key_prefix)])

    return log_statement, stats_dict

def ignore_None_collate(batch):
    '''
    dataloader.default_collate wrapper that creates batches only of not None values.
    Useful for cases when the dataset.__getitem__ can return None because of some
    exception and then we will want to exclude that sample from the batch.
    '''
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return data.dataloader.default_collate(batch)
