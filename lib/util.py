import os
import sys
import math
import importlib
import numpy as np
import shutil
import logging
import datetime
import pprint
from time import *
from glob import glob

def copyfile(src, dst):
    shutil.copyfile(src, dst)


def pretty_print(name, input, val_width=40, key_width=0):
    """
    This function creates a formatted string from a given dictionary input.
    It may not support all data types, but can probably be extended.

    Args:
        name (str): name of the variable root
        input (dict): dictionary to print
        val_width (int): the width of the right hand side values
        key_width (int): the minimum key width, (always auto-defaults to the longest key!)

    Example:
        pretty_str = pretty_print('conf', conf.__dict__)
        pretty_str = pretty_print('conf', {'key1': 'example', 'key2': [1,2,3,4,5], 'key3': np.random.rand(4,4)})

        print(pretty_str)
        or
        logging.info(pretty_str)
    """

    # root
    pretty_str = name + ': {\n'

    # determine key width
    for key in input.keys(): key_width = max(key_width, len(str(key)) + 4)

    # cycle keys
    for key in input.keys():

        val = input[key]

        # round values to 3 decimals..
        if type(val) == np.ndarray: val = np.round(val, 3).tolist()

        # difficult formatting
        val_str = str(val)
        if len(val_str) > val_width:
            val_str = pprint.pformat(val, width=val_width, compact=True)
            val_str = val_str.replace('\n', '\n{tab}')
            tab = ('{0:' + str(4 + key_width) + '}').format('')
            val_str = val_str.replace('{tab}', tab)

        # more difficult formatting
        format_str = '{0:' + str(4) + '}{1:' + str(key_width) + '} {2:' + str(val_width) + '}\n'
        pretty_str += format_str.format('', key + ':', val_str)

    # close root object
    pretty_str += '}'

    return pretty_str


def convertAlpha2Rot_torch(alpha, z3d, x3d):

    ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi
    # ry3d = alpha + torch.atan2(x3d, z3d)# + 0.5 * math.pi

    return ry3d

def absolute_import(file_path):
    """
    Imports a python module / file given its ABSOLUTE path.

    Args:
         file_path (str): absolute path to a python file to attempt to import
    """

    # module name
    _, name, _ = file_parts(file_path)

    # load the spec and module
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module



def file_parts(file_path):
    """
    Lists a files parts such as base_path, file name and extension

    Example
        base, name, ext = file_parts('path/to/file/dog.jpg')
        print(base, name, ext) --> ('path/to/file/', 'dog', '.jpg')
    """

    base_path, tail = os.path.split(file_path)
    name, ext = os.path.splitext(tail)

    return base_path, name, ext

def compute_eta(start_time, idx, total):
    """
    Computes the estimated time as a formatted string as well
    as the change in delta time dt.

    Example:
        from time import time

        start_time = time()

        for i in range(0, total):
            <lengthly computation>
            time_str, dt = compute_eta(start_time, i, total)
    """

    dt = (time() - start_time)/idx
    timeleft = np.max([dt * (total - idx), 0])
    if timeleft > 3600: time_str = '{:.1f}h'.format(timeleft / 3600);
    elif timeleft > 60: time_str = '{:.1f}m'.format(timeleft / 60);
    else: time_str = '{:.1f}s'.format(timeleft);

    return time_str, dt

def convertRot2Alpha(ry3d, z3d, x3d):

    alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
    # alpha = ry3d - math.atan2(x3d, z3d)  # equivalent

    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha

def list_files(base_dir, file_pattern):
    """
    Returns a list of files given a directory and pattern
    The results are sorted alphabetically

    Example:
        files = list_files('path/to/images/', '*.jpg')
    """

    return sorted(glob(os.path.join(base_dir) + file_pattern))

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """

    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)

def init_log_file(folder_path, suffix=None, log_level=logging.INFO):
    """
    This function inits a log file given a folder to write the log to.
    it automatically adds a timestamp and optional suffix to the log.
    Anything written to the log will automatically write to console too.

    Example:
        import logging

        init_log_file('output/logs/')
        logging.info('this will show up in both the log AND console!')
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = '[%(levelname)s]: %(asctime)s %(message)s'

    if suffix is not None:
        file_name = timestamp + '_' + suffix
    else:
        file_name = timestamp

    logging.root.handlers = []
    file_path = os.path.join(folder_path, file_name + '.log')
    logging.basicConfig(filename=file_path, level=log_level, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(__name__)
    logger.info(file_path)

    return logger

def convertAlpha2Rot(alpha, z3d, x3d):

    ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi
    #ry3d = alpha + math.atan2(x3d, z3d)# + 0.5 * math.pi

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return ry3d