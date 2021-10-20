import pickle
import pdb

NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}
LOAD_METADATA_STR = "Loading risk trajectory dataset from {}..."

def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]

# Depending on arg, build dataset
def get_dataset(args):
    dataset_class = get_dataset_class(args)

    print(LOAD_METADATA_STR.format(args.metadata_pickle_path))
    metadata = pickle.load(open(args.metadata_pickle_path,'rb'))

    train = dataset_class(args, metadata, 'train')
    dev = dataset_class(args, metadata, 'dev')
    test = dataset_class(args, metadata, 'test')

    return train, dev, test
