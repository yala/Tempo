import torch
from torch import nn
import pdb

MODEL_REGISTRY = {}

NO_MODEL_ERR = 'Model {} not in MODEL_REGISTRY! Available models are {} '
NO_OPTIM_ERR = 'Optimizer {} not supported!'
NO_OPTIM_FOR_NO_PARAM_MSG = "No optoimizer created for model since it has no parameters"

def RegisterModel(model_name):
    """Registers a configuration."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator

def get_model(args):
    models = {}
    if args.progression_snapshot is None:
        models['progression'] = get_model_by_name(args.progression_model_name, args)
    else:
        models['progression'] = load_model(args.progression_snapshot, args)
        if isinstance(models['progression'], MODEL_REGISTRY['gru_w_cum_hazard']):
            args.prog_hidden_dim = models['progression'].args.hidden_dim * models['progression'].args.num_layers
        else:
            args.prog_hidden_dim = models['progression'].args.hidden_dim

    if args.screening_snapshot is None:
        models['screening'] = get_model_by_name(args.screening_model_name, args)
    else:
        models['screening'] = load_model(args.screening_snapshot, args)

    models['model'] = models[args.task]

    for _, model in models.items():
        if hasattr(model, 'args'):
            model.args.envelope_inference = args.envelope_inference
            model.args.max_steps = args.max_steps
    return models

def get_model_by_name(name, args):
    '''
        Get model from MODEL_REGISTRY based on args.model_name
        args:
        - name: Name of model, must exit in registry
        - allow_wrap_model: whether or not override args.wrap_model and disable model_wrapping.
        - args: run ime args from parsing

        returns:
        - model: an instance of some torch.nn.Module
    '''
    if not name in MODEL_REGISTRY:
        raise Exception(
            NO_MODEL_ERR.format(
                name, MODEL_REGISTRY.keys()))


    model = MODEL_REGISTRY[name](args)
    return wrap_model(model, args)

def wrap_model(model, args):

    if args.num_gpus > 1 and args.data_parallel and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model,
                                    device_ids=range(args.num_gpus))
    return model

def load_model(path, args, do_wrap_model = True):
    print('\nLoading model from [%s]...' % path)
    try:
        model = torch.load(path, map_location='cpu')
        if isinstance(model, dict):
            model = model['model']

        if isinstance(model, nn.DataParallel):
            model = model.module.cpu()
        model = wrap_model(model, args)
    except:
        raise Exception(
            "Sorry, snapshot {} does not exist!".format(path))
    return model


def get_params(model):
    '''
    Helper function to get parameters of a model.
    '''

    return model.parameters()


def get_optimizer(model, args):
    '''
    Helper function to fetch optimizer based on args.
    '''
    params = [param for param in model.parameters() if param.requires_grad]
    if len(params) == 0:
        print(NO_OPTIM_FOR_NO_PARAM_MSG)
        return None
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        return torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(params,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum )
    else:
        raise Exception(NO_OPTIM_ERR.format(args.optimizer))

