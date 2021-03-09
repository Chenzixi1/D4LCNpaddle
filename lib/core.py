import paddle.fluid as fluid
from easydict import EasyDict as edict

from lib.util import *

def init_config(conf_name):
    """
    Loads configuration file, by checking for the conf_name.py configuration file as
    ./config/<conf_name>.py which must have function "Config".

    This function must return a configuration dictionary with any necessary variables for the experiment.
    """

    conf = importlib.import_module('config.' + conf_name).Config()

    return conf


def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter

        # torch.Tensor
        if data_type == fluid.Tensor:
            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))


def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # fluid.Variable
        if data_type == fluid.core_avx.VarBase:
            max_xy = fluid.layers.elementwise_min(box_a[:, 2:], box_b[:, 2:])
            min_xy = fluid.layers.elementwise_max(box_a[:, :2], box_b[:, :2])
            inter = fluid.layers.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.max(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou_ign(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of overap of box_b has within box_a, which is handy for dealing with ignore regions.
    Hence, assume that box_b are ignore regions and box_a are anchor boxes, then we may want to know how
    much overlap the anchors have inside of the ignore regions (hence ignore area_b!)
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) * 0 - inter * 0

        # torch and numpy have different calls for transpose
        if data_type == fluid.Tensor:
            return (inter / union).permute(1, 0)
        elif data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

    else:
        raise ValueError('unknown mode {}'.format(mode))

def adjust_lr(conf, optimizer, iter):
    """
    Adjusts the learning rate of an optimizer according to iteration and configuration,
    primarily regarding regular SGD learning rate policies.

    Args:
        conf (dict): configuration dictionary
        optimizer (object): paddle optim object
        iter (int): current iteration
    """

    if 'batch_skip' in conf and ((iter + 1) % conf.batch_skip) > 0: return

    if conf.solver_type.lower() == 'sgd':

        lr = conf.lr
        lr_steps = conf.lr_steps
        max_iter = conf.max_iter
        lr_policy = conf.lr_policy
        lr_target = conf.lr_target

        if lr_steps:
            steps = np.array(lr_steps) * max_iter
            total_steps = steps.shape[0]
            step_count = np.sum((steps - iter) <= 0)

        else:
            total_steps = max_iter
            step_count = iter

        # perform the exact number of steps needed to get to lr_target
        if lr_policy.lower() == 'step':
            scale = (lr_target / lr) ** (1 / total_steps)
            lr *= scale ** step_count

            # update the actual learning rate
            for gind, g in enumerate(optimizer._parameter_list):
                g['lr'] = lr
        # compute the scale needed to go from lr --> lr_target
        # using a polynomial function instead.
        elif lr_policy.lower() == 'poly':

            power = 0.9
            scale = total_steps / (1 - (lr_target / lr) ** (1 / power))
            lr *= (1 - step_count / scale) ** power

            # update the actual learning rate
            for gind, g in enumerate(optimizer._parameter_list):
                g['lr'] = lr

        # elif lr_policy.lower() == 'cosinerestart':
        #     scheduler.step()
        #
        # elif lr_policy.lower() == 'onecycle':
        #     scheduler.step()
        #
        # elif lr_policy.lower() == 'cosinepoly':
        #     scheduler.step()

            power = 0.9
            scale = total_steps / (1 - (lr_target / lr) ** (1 / power))
            lr_weight = (1 - step_count / scale) ** power
            for gind, g in enumerate(optimizer._parameter_list):
                g['lr'] = g['lr'] * lr_weight
                # print(g['lr'])

def init_training_paths(conf_name, result_dir, use_tmp_folder=None):
    """
    Simple function to store and create the relevant paths for the project,
    based on the base = current_working_dir (cwd). For this reason, we expect
    that the experiments are run from the root folder.

    data    =  ./data
    output  =  ./output/<conf_name>
    weights =  ./output/<conf_name>/weights
    results =  ./output/<conf_name>/results
    logs    =  ./output/<conf_name>/log

    Args:
        conf_name (str): configuration experiment name (used for storage into ./output/<conf_name>)
    """

    # make paths
    paths = edict()
    paths.base = os.getcwd()
    paths.data = os.path.join(paths.base, 'data')
    paths.output = os.path.join(os.getcwd(), 'output', conf_name)
    paths.weights = os.path.join(paths.output, result_dir, 'weights')
    paths.logs = os.path.join(paths.output, result_dir, 'log')

    if use_tmp_folder:
        paths.results = os.path.join(paths.base, '.tmp_results', conf_name, 'results')
    else:
        paths.results = os.path.join(paths.output, result_dir, 'results')

    # make directories
    mkdir_if_missing(paths.output)
    mkdir_if_missing(paths.logs)
    mkdir_if_missing(paths.weights)
    mkdir_if_missing(paths.results)

    return paths



def load_weights(model, path, remove_module=False):
    """
    Simply loads a pytorch models weights from a given path.
    """
    dst_weights = model.state_dict()
    src_weights, _ = fluid.load_dygraph(path)


    dst_keys = list(dst_weights.keys())
    src_keys = list(src_weights.keys())

    if remove_module:

        # copy keys without module
        for key in src_keys:
            src_weights[key.replace('module.', '')] = src_weights[key]
            del src_weights[key]
        src_keys = list(src_weights.keys())

        # remove keys not in dst
        for key in src_keys:
            if key not in dst_keys: del src_weights[key]

    else:

        # remove keys not in dst
        for key in src_keys:
            if key not in dst_keys: del src_weights[key]

        # add keys not in src
        for key in dst_keys:
            if key not in src_keys: src_weights[key] = dst_weights[key]

    model.load_dict(src_weights)

def resume_checkpoint(optim, model, weights_dir, iteration):
    """
    Loads the optimizer and model pair given the current iteration
    and the weights storage directory.
    """

    optimpath, modelpath = checkpoint_names(weights_dir, iteration)

    optim.load_dict(fluid.load_dygraph(optimpath))
    model.load_dict(fluid.load_dygraph(modelpath))

def checkpoint_names(weights_dir, iteration):
    """
    Single function to determine the saving format for
    resuming and saving models/optim.
    """

    optimpath = os.path.join(weights_dir, 'optim_{}_pkl'.format(iteration))
    modelpath = os.path.join(weights_dir, 'model_{}_pkl'.format(iteration))

    return optimpath, modelpath

def freeze_layers(network, blacklist=None, whitelist=None, verbose=False):

    if blacklist is not None:

        for name, param in network.named_parameters():

            if not any([allowed in name for allowed in blacklist]):
                if verbose:
                    logging.info('freezing {}'.format(name))
                param.requires_grad = False

        for name, module in network.named_modules():
            if not any([allowed in name for allowed in blacklist]):
                if isinstance(module, fluid.dygraph.BatchNorm):
                    module.eval()

    if whitelist is not None:

        for name, param in network.named_parameters():

            if any([banned in name for banned in whitelist]):
                if verbose:
                    logging.info('freezing {}'.format(name))
                param.requires_grad = False
            #else:
            #    logging.info('NOT freezing {}'.format(name))

        for name, module in network.named_modules():
            if any([banned in name for banned in whitelist]):
                if isinstance(module, fluid.dygraph.BatchNorm):
                    module.eval()

def compute_stats(tracker, stats):
    """
    Copies any arbitary statistics which appear in 'stats' into 'tracker'.
    Also, for each new object to track we will secretly store the objects information
    into 'tracker' with the key as (group + name + '_obj'). This way we can retrieve these properties later.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        stats (array): dictionary array tracker objects. See below.

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    # through all stats
    for stat in stats:

        # get properties
        name = stat['name']
        group = stat['group']
        val = stat['val']

        # convention for identificaiton
        id = group + name

        # init if not exist?
        if not (id in tracker): tracker[id] = []

        # convert tensor to numpy
        if type(val) == fluid.core_avx.VarBase:
            val = val.detach().numpy()

        # store
        tracker[id].append(val)

        # store object info
        obj_id = id + '_obj'
        if not (obj_id in tracker):
            stat.pop('val', None)
            tracker[id + '_obj'] = stat


def log_stats(tracker, iteration, start_time, start_iter, max_iter, skip=1):
    """
    This function writes the given stats to the log / prints to the screen.
    Also, computes the estimated time arrival (eta) for completion and (dt) delta time per iteration.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        iteration (int): the current iteration
        start_time (float): starting time of whole experiment
        start_iter (int): starting iteration of whole experiment
        max_iter (int): maximum iteration to go to

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    display_str = 'iter: {}'.format((int(iteration/skip)))

    # compute eta
    time_str, dt = compute_eta(start_time, iteration - start_iter, max_iter - start_iter)

    # cycle through all tracks
    last_group = ''
    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:

            # compute mean
            meanval = np.mean(tracker[key])

            # get properties
            format = tracker[key + '_obj'].format
            group = tracker[key + '_obj'].group
            name = tracker[key + '_obj'].name

            # logic to have the string formatted nicely
            # basically roughly this format:
            #   iter: {}, group_1 (name: val, name: val), group_2 (name: val), dt: val, eta: val
            if last_group != group and last_group == '':
                display_str += (', {} ({}: ' + format).format(group, name, meanval)

            elif last_group != group:
                display_str += ('), {} ({}: ' + format).format(group, name, meanval)

            else:
                display_str += (', {}: ' + format).format(name, meanval)

            last_group = group

    # append dt and eta
    display_str += '), dt: {:0.2f}, eta: {}'.format(dt, time_str)

    # log
    logging.info(display_str)

def Visual_stats(tracker, iteration, writer):

    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:
            if 'acc' in key:
                value = np.mean(tracker[key])
                tag = "acc/" + key[3:]
                writer.add_scalar(tag=tag, step=iteration, value=value)

            if 'loss' in key:
                value = np.mean(tracker[key])
                tag = "loss/" + key[4:]
                writer.add_scalar(tag=tag, step=iteration, value=value)

            if 'misc' in key:
                value = np.mean(tracker[key])
                tag = "misc/" + key[4:]
                writer.add_scalar(tag=tag, step=iteration, value=value)


def save_checkpoint(optim, model, weights_dir, iteration):
    """
    Saves the optimizer and model pair given the current iteration
    and the weights storage directory.
    """

    optimpath, modelpath = checkpoint_names(weights_dir, iteration)

    fluid.dygraph.save_dygraph(model.state_dict(), modelpath)
    # fluid.dygraph.save_dygraph(optim.state_dict(), optimpath)