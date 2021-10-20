import os
import math
import numpy as np
import sklearn.metrics
import torch
from tqdm import tqdm
import oncopolicy.models.factory as model_factory
import oncopolicy.learn.state_keeper as state
from oncopolicy.utils.learn import ignore_None_collate, init_metrics_dictionary, \
    get_train_and_dev_dataset_loaders, collate_eval_metrics
from collections import defaultdict
from oncopolicy.learn.step import progression_model_step, screening_model_step
from oncopolicy.utils.stats import confidence_interval
import warnings

tqdm.monitor_interval=0

MODELS_WITH_OPTIMIZERS = ['model']

def get_train_variables(args, models):
    '''
        Given args, and whether or not resuming training, return
        relevant train variales.

        returns:
        - start_epoch:  Index of initial epoch
        - epoch_stats: Dict summarizing epoch by epoch results
        - state_keeper: Object responsibile for saving and restoring training state
        - batch_size: sampling batch_size
        - models: Dict of models
        - optimizers: Dict of optimizers, one for each model
        - tuning_key: Name of epoch_stats key to control learning rate by
        - num_epoch_sans_improvement: Number of epochs since last dev improvment, as measured by tuning_key
        - num_epoch_since_reducing_lr: Number of epochs since last lr reduction
        - no_tuning_on_dev: True when training does not adapt based on dev performance
    '''
    start_epoch = 1
    if args.current_epoch is not None:
        start_epoch = args.current_epoch
    if args.lr is None:
        args.lr = args.init_lr
    if args.epoch_stats is not None:
        epoch_stats = args.epoch_stats
    else:
        epoch_stats = init_metrics_dictionary()

    state_keeper = state.StateKeeper(args)
    batch_size = args.batch_size // args.batch_splits

    assert isinstance(models, dict) and 'model' in models

    for name, indiv_model in models.items(): 
        models[name] = indiv_model.to(args.device)
    # Setup optimizers
    optimizers = {}
    for name, indiv_model in models.items(): 
        if name in MODELS_WITH_OPTIMIZERS:
            optimizers[name] = model_factory.get_optimizer(indiv_model, args)

    if args.optimizer_state is not None:
        for optimizer_name in args.optimizer_state:
            state_dict = args.optimizer_state[optimizer_name]
            optimizers[optimizer_name] = state_keeper.load_optimizer(
                optimizers[optimizer_name],
                state_dict)

    num_epoch_sans_improvement = 0
    num_epoch_since_reducing_lr = 0

    no_tuning_on_dev = args.no_tuning_on_dev

    tuning_key = "dev_{}".format(args.tuning_metric)

    return start_epoch, epoch_stats, state_keeper, batch_size, models, optimizers, tuning_key, num_epoch_sans_improvement, num_epoch_since_reducing_lr, no_tuning_on_dev


def train_model(train_data, dev_data, model, args):
    '''
        Train model and tune on dev set. If model doesn't improve dev performance within args.patience
        epochs, then halve the learning rate, restore the model to best and continue training.

        At the end of training, the function will restore the model to best dev version.

        returns epoch_stats: a dictionary of epoch level metrics for train and test
        returns models : dict of models, containing best performing model setting from this call to train
    '''

    start_epoch, epoch_stats, state_keeper, batch_size, models, optimizers, tuning_key, num_epoch_sans_improvement, num_epoch_since_reducing_lr, no_tuning_on_dev = get_train_variables(
        args, model)

    train_data_loader, dev_data_loader = get_train_and_dev_dataset_loaders(
        args,
        train_data,
        dev_data,
        batch_size)

    for epoch in range(start_epoch, args.epochs + 1):

        print("-------------\nEpoch {}:\n".format(epoch))

        for mode, data_loader in [('Train', train_data_loader), ('Dev', dev_data_loader)]:
            train_model = mode == 'Train'
            key_prefix = mode.lower()
            loss, preds, ssns, exams, epoch_metrics = run_epoch(
                data_loader,
                train_model=train_model,
                truncate_epoch=True,
                models=models,
                optimizers=optimizers,
                args=args)

            log_statement, epoch_stats = collate_eval_metrics(
                                args, loss, preds, ssns,
                                exams, epoch_metrics,
                                epoch_stats, key_prefix)

            print(log_statement)

        # Save model if beats best dev, or if not tuning on dev
        best_func, arg_best = (min, np.argmin) if 'loss' in tuning_key else (max, np.argmax)
        print("Doing model selection to pick the {} model for {}".format(best_func, arg_best))
        improved = best_func(epoch_stats[tuning_key]) == epoch_stats[tuning_key][-1]
        if improved or no_tuning_on_dev:
            num_epoch_sans_improvement = 0
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            epoch_stats['best_epoch'] = arg_best( epoch_stats[tuning_key] )
            state_keeper.save(models, optimizers, epoch, args.lr, epoch_stats)

        num_epoch_since_reducing_lr += 1
        if improved:
            num_epoch_sans_improvement = 0
        else:
            num_epoch_sans_improvement += 1
        print('---- Best Dev {} is {} at epoch {}'.format(
            args.tuning_metric,
            epoch_stats[tuning_key][epoch_stats['best_epoch']],
            epoch_stats['best_epoch'] + 1))

        if num_epoch_sans_improvement >= args.patience or \
                (no_tuning_on_dev and num_epoch_since_reducing_lr >= args.lr_reduction_interval):
            print("Reducing learning rate")
            num_epoch_sans_improvement = 0
            num_epoch_since_reducing_lr = 0
            if not args.turn_off_model_reset:
                models, optimizer_states, _, _, _ = state_keeper.load()

                # Reset optimizers
                for name in optimizers:
                    optimizer = optimizers[name]
                    state_dict = optimizer_states[name]
                    optimizers[name] = state_keeper.load_optimizer(optimizer, state_dict)
            # Reduce LR
            for name in optimizers:
                optimizer = optimizers[name]
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay

            # Update lr also in args for resumable usage
            args.lr *= .5

    # Restore model to best dev performance, or last epoch when not tuning on dev
    models, _, _, _, _ = state_keeper.load()

    return epoch_stats, models

def eval_model(test_data, models, key_prefix, args):
    '''
        Run model on test data, and return test stats (includes loss

        accuracy, etc)
    '''
    for name in models:
        models[name] = models[name].to(args.device)

    batch_size = args.batch_size // args.batch_splits
    test_stats = init_metrics_dictionary()
    data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False)

    loss, preds, ssns, exams, epoch_metrics = run_epoch(
        data_loader,
        train_model=False,
        truncate_epoch=False,
        models=models,
        optimizers=None,
        args=args)

    log_statement, test_stats = collate_eval_metrics(
                            args, loss, preds, ssns,
                            exams, epoch_metrics,
                            test_stats, key_prefix)
    print(log_statement)

    return test_stats

def run_epoch(data_loader, train_model, truncate_epoch, models, optimizers, args):
    '''
        Run model for one pass of data_loader, and return epoch statistics.
        args:
        - data_loader: Pytorch dataloader over some dataset.
        - train_model: True to train the model and run the optimizers
        - models: dict of models, where 'model' is the main model, and others can be critics, or meta-models
        - optimizer: dict of optimizers, one for each model
        - args: general runtime args defined in by argparse
        returns:
        - avg_loss: epoch loss
        - golds: labels for all samples in data_loader
        - preds: model predictions for all samples in data_loader
        - probs: model softmaxes for all samples in data_loader
        - exams: exam ids for samples if available, used to cluster samples for evaluation.
    '''
    data_iter = data_loader.__iter__()
    preds = []
    losses = []
    aux_metrics = defaultdict(list)
    ssns = []
    exams = []
    has_ca = []

    torch.set_grad_enabled(train_model)
    for name in models:
        if train_model:
            models[name].train()
            if optimizers is not None and name in optimizers:
                optimizers[name].zero_grad()
        else:
            models[name].eval()

    batch_loss = 0
    num_batches_per_epoch = len(data_loader)

    if truncate_epoch:
        max_batches =  args.max_batches_per_train_epoch if train_model else args.max_batches_per_dev_epoch
        num_batches_per_epoch = min(len(data_loader), (max_batches * args.batch_splits))

    num_steps = 0
    i = 0
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch)
    for batch in data_iter:
        if batch is None:
            warnings.warn('Empty batch')
            continue

        x, batch = prepare_batch(batch, args)

        if args.task == 'progression':
            model_step = progression_model_step
        else:
            model_step = screening_model_step
            assert args.task ==  'screening'

        step_results = model_step(x, batch, models, optimizers, train_model, args)
        loss, batch_metrics, batch_preds, batch_ssns, batch_exams, batch_has_ca = step_results

        batch_loss += loss.item()

        if train_model:
            if (i + 1) % args.batch_splits == 0:
                optimizers['model'].step()
                optimizers['model'].zero_grad()

            if (i+1) % args.reset_rate == 0 and args.task ==  'screening':
                models['model'].reset()


        if (i + 1) % args.batch_splits == 0:
            losses.append(batch_loss)
            batch_loss = 0

        ssns.extend(batch_ssns)
        exams.extend(batch_exams)
        preds.extend(batch_preds)
        has_ca.extend(batch_has_ca)

        for key in batch_metrics:
            batch_metric_value = batch_metrics[key]
            if isinstance(batch_metric_value, list):
                aux_metrics[key].extend( batch_metric_value )
            else:
                aux_metrics[key].append( batch_metric_value )

        i += 1
        num_steps += 1
        tqdm_bar.update()
        if i > num_batches_per_epoch:
            data_iter.__del__()
            break

    avg_loss = np.mean(losses)
    num_batches = len(losses)
    num_ca = sum(has_ca)
    metric_keys = list(aux_metrics.keys())
    for key in metric_keys:
        if 'cancer' in key:
            # Compute cancer metrics on cancer exams
            emperical_distribution = [metric for metric, is_cancer in zip(aux_metrics[key], has_ca) if is_cancer]
            emperical_ssns = [ssn for ssn, is_cancer in zip(ssns, has_ca) if is_cancer]
        else:
            emperical_distribution = aux_metrics[key]
            emperical_ssns = ssns
        estimator = np.mean

        if args.get_conf_intervals and len(emperical_distribution) == len(emperical_ssns):
            mean_estimate, (lower_bound, upper_bound) = confidence_interval(emperical_distribution, estimator=estimator, clusters=emperical_ssns)
            aux_metrics[key] = mean_estimate
            aux_metrics["{}_lower_95".format(key)] = lower_bound
            aux_metrics["{}_upper_95".format(key)] = upper_bound
        else:
            aux_metrics[key] = estimator(emperical_distribution)
        aux_metrics['{}_list'.format(key)] = emperical_distribution
        aux_metrics['has_cancer_list'] = has_ca

    return avg_loss, preds, ssns, exams, aux_metrics


def prepare_batch(batch, args):
    x = batch['x'].to(args.device)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = batch[key].to(args.device)

    return x, batch
