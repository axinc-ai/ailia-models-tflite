import os
import sys
import argparse
import glob
from logging import DEBUG

from params import MODALITIES, EXTENSIONS
import log_init

# FIXME: Next two lines should be better to call from the main script
# once we prepared one. For now, we do the initialization of logger here.
logger = log_init.logger
logger.info('Start!')

# TODO: better to use them (first, fix above)
# from logging import getLogger
# logger = getLogger(__name__)


# TODO: yaml config file and yaml loader

try:
    import ailia_tflite
    AILIA_EXIST = True
except ImportError:
    logger.warning('ailia_tflite package cannot be found under `sys.path`')
    AILIA_EXIST = False

def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        logger.error(f'{filename} not found')
        sys.exit()


def get_base_parser(
        description, default_input, default_save, input_ftype='image',
):
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
        help=('The default (model-dependent) input data (image / video) path. '
              'If a directory name is specified, the model will be run for '
              'the files inside. File type is specified by --ftype argument')
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
        '--ftype', metavar='FILE_TYPE', default=input_ftype,
        choices=MODALITIES,
        help='file type list: ' + ' | '.join(MODALITIES)
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='set default logger level to DEBUG (enable to show DEBUG logs)'
    )
    parser.add_argument(
        '--profile', action='store_true',
        help='set profile mode (enable to show PROFILE logs)'
    )
    parser.add_argument(
        '-bc', '--benchmark_count', metavar='BENCHMARK_COUNT',
        default=5, type=int,
        help='set iteration count of benchmark'
    )
    parser.add_argument(
        '--tflite', action='store_true',
        help='By default, the ailia lite runtime is used, but with this ' +
        'option, you can switch to using the TensorFlow Lite runtime.'
    )
    parser.add_argument(
        '-f', '--flags', metavar='FLAGS',
        default=0, type=int,
        help='set flags for tflite runtime'
    )
    parser.add_argument(
        '-m', '--memory_mode', metavar='MEMORY_MODE',
        default=0, type=int,
        help='set memory mode for tflite runtime'
    )
    parser.add_argument(
        '--shape', type=int, 
        help='change input image shape (Please specify one int value to change width and height).'
    )
    parser.add_argument(
        '--float', action='store_true',
        help='use float model.'
    )
    parser.add_argument(
        '-e', '--env_id', type=int, default=0,
        help='set environment id. 0 = CPU, 1 = NNAPI, 2 = MMALIB, 3 = MMALIB_COMPATIBLE, 4 = QNN, 5 = DELEGATE'
    )
    parser.add_argument(
        '-cw', '--camera_width',
        default=0, type=int,
        help='set camera width'
    )
    parser.add_argument(
        '-ch', '--camera_height',
        default=0, type=int,
        help='set camera height'
    )
    parser.add_argument(
        '--delegate_path', type=str, default=None,
        help='external delegate file path'
    )
    return parser


def update_parser(parser, check_input_type=True):
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

    # -------------------------------------------------------------------------
    # 0. logger level update
    if args.debug:
        logger.setLevel(DEBUG)

    # -------------------------------------------------------------------------
    # 1. update input
    if args.video is not None:
        args.ftype = 'video'
        args.input = None # force video mode

    if args.input is None:
        # TODO: args.video, args.input is vague...
        # input is None --> video mode maybe?
        pass
    elif isinstance(args.input, list):
        # LIST --> nothing will be changed here.
        pass
    elif os.path.isdir(args.input):
        # Directory Path --> generate list of inputs
        files_grapped = []
        in_dir = args.input
        for extension in EXTENSIONS[args.ftype]:
            files_grapped.extend(glob.glob(os.path.join(in_dir, extension)))
        logger.info(f'{len(files_grapped)} {args.ftype} files found!')

        args.input = sorted(files_grapped)

        # create save directory
        if args.savepath is None:
            pass
        else:
            if '.' in args.savepath:
                logger.warning('Please specify save directory as --savepath '
                               'if you specified a direcotry for --input')
                logger.info(f'[{in_dir}_results] directory will be created')
                if in_dir[-1] == '/':
                    in_dir = in_dir[:-1]
                args.savepath = in_dir + '_results'
            os.makedirs(args.savepath, exist_ok=True)
            logger.info(f'output saving directory: {args.savepath}')

    elif os.path.isfile(args.input):
        args.input = [args.input]
    else:
        if check_input_type:
            logger.error('specified input is not file path nor directory path')
            sys.exit(0)

    # -------------------------------------------------------------------------
    return args

def get_savepath(arg_path, src_path, prefix='', post_fix='_res', ext=None):
    """Get savepath
    NOTE: we may have better option...
    TODO: args.save_dir & args.save_path ?

    Parameters
    ----------
    arg_path : str
        argument parser's savepath
    src_path : str
        the path of source path
    prefix : str, default is ''
    postfix : str, default is '_res'
    ext : str, default is None
        if you need to specify the extension, use this argument
        the argument has to start with '.' like '.png' or '.jpg'

    Returns
    -------
    new_path : str
    """

    if '.' in arg_path:
        # 1. args.savepath is actually the image path
        arg_base, arg_ext = os.path.splitext(arg_path)
        new_ext = arg_ext if ext is None else ext
        new_path = arg_base + new_ext
    else:
        # 2. args.savepath is save directory path
        src_base, src_ext = os.path.splitext(os.path.basename(src_path))
        new_ext = src_ext if ext is None else ext
        new_path = os.path.join(
            arg_path, prefix + src_base + post_fix + new_ext
        )
    return new_path

def delegate_obj(delegate_path):
    if delegate_path is None:
        return None
    else:
        return [ailia_tflite.load_delegate(delegate_path)]
