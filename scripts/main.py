import pickle
from os.path import dirname, realpath
import sys
import git
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import oncopolicy.datasets.factory as dataset_factory
import oncopolicy.models.factory as model_factory
from oncopolicy.learn import train
import oncopolicy.utils.parsing as parsing
import warnings
import oncopolicy.learn.state_keeper as state
import copy

#Constants
DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"

if __name__ == '__main__':
    args = parsing.parse_args()

    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    print("OncoPolicy main running from commit: \n\n{}\n{}author: {}, date: {}".format(
        commit.hexsha, commit.message, commit.author, commit.committed_date))

    # Load dataset and add dataset specific information to args
    print("\nLoading data...")
    train_data, dev_data, test_data = dataset_factory.get_dataset(args)

    # Load model and add model specific information to args
    model = model_factory.get_model(args)

    print(model)
    # Load run parameters if resuming that run.
    args.model_path = state.get_model_path(args)
    print('Trained models will be saved to [%s]' % args.model_path)

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state']:
            print("\t{}={}".format(attr.upper(), value))

    save_path = args.results_path
    print()
    if args.train:
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        args.epoch_stats = epoch_stats

        print("Save train/dev results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    print()
    if args.dev:
        print("-------------\nDev")
        args.dev_stats = train.eval_model(dev_data, model, 'dev', args)
        print("Save dev results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if args.test:
        print("-------------\nTest")
        args.test_stats = train.eval_model(test_data, model, 'test', args)
        print("Save test results to {}".format(save_path))
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if args.do_subgroup_eval and ("mgh" in args.metadata_pickle_path or 'emory' in args.metadata_pickle_path):
        print("----------\n Subgroup Eval on Test")
        subgroup_lambdas = parsing.load_subgroups(args)
        filtered_data = copy.deepcopy(test_data)
        args_dict = vars(args)
        for name, filt_fn in subgroup_lambdas.items():
            print("Evaluating: {}".format(name))
            filtered_data.dataset =  [sample for sample in test_data.dataset if filt_fn(sample)]
            print("{} dataset has {} exams from {} patients. {} patients develop cancer".format(name, len(filtered_data.dataset), len( set( [d['ssn'] for d in filtered_data.dataset])), len( set( [d['ssn'] for d in filtered_data.dataset if d['ever_has_cancer']] ))) )
            if len(filtered_data) == 0:
                print("Skip {} because no matches".format(name))
                continue
            args_dict['{}_test_stats'.format(name)] = train.eval_model(filtered_data, model, 'test', args)
        pickle.dump(args_dict, open(save_path, 'wb'))


