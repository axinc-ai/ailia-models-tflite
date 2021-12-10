import os
import sys
import argparse

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)

def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        print(f'[ERROR] {filename} not found')
        sys.exit()


def get_base_parser(description, default_input, default_save, parse=False):
    """
    Get ailia default argument parser

    Parameters
    ----------
    description : str
    default_input : str
        default input data (image / video) path
    default_save : str
        default save path
    parse : bool, default is False
        if True, return parsed arguments
        TODO: deprecates

    Returns
    -------
    out : ArgumentParser()

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description,
        conflict_handler='resolve',  # allow to overwrite default argument
    )
    parser.add_argument(
        '-i', '--input', metavar='IMAGE/VIDEO', default=default_input,
        help='The default (model-dependent) input data (image / video) path.'
    )
    parser.add_argument(
        '-v', '--video', metavar='VIDEO', default=None,
        help=('You can convert the input video by entering style image.'
              'If the int variable is given, '
              'corresponding webcam input will be used.')
    )
    parser.add_argument(
        '-s', '--savepath', metavar='SAVE_PATH', default=default_save,
        help='Save path for the output (image / video).'
    )
    parser.add_argument(
        '-b', '--benchmark', action='store_true',
        help=('Running the inference on the same input 5 times to measure '
              'execution performance. (Cannot be used in video mode)')
    )
    parser.add_argument(
        '-t', '--tflite', action='store_true',
        help='By default, the ailia lite runtime is used, but with this ' +
        'option, you can switch to using the TensorFlow Lite runtime.'
    )

    if parse:
        parser = parser.parse_args()

    return parser


def update_parser(parser):
    """Default check or update configurations should be placed here

    Parameters
    ----------
    parser : ArgumentParser()

    Returns
    -------
    args : ArgumentParser()
        (parse_args() will be done here)
    """
    args = parser.parse_args()

    return args
