import os
import urllib.request


def progress_print(block_count, block_size, total_size):
    """
    Callback function to display the progress
    (ref: https://qiita.com/jesus_isao/items/ffa63778e7d3952537db)

    Parameters
    ----------
    block_count:
    block_size:
    total_size:
    """
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        # Bigger than 100 does not look good, so...
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = '=' * bar_num
    if bar_num != max_bar:
        progress_element += '>'
    bar_fill = ' '  # fill the blanks
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(f'[{bar} {percentage:.2f}% ( {total_size_kb:.0f}KB )]', end='\r')


def check_and_download_models(model_path, remote_path):
    """
    Check if the tflite file exists, and if necessary, download the file to the
    given path.

    Parameters
    ----------
    model_path: string
        The path of tflite file for ailia.
    remote_path: string
        The url where the tflite file is saved.
        ex. "https://storage.googleapis.com/ailia-models-tflite/mobilenetv2/"
    """

    if not os.path.exists(model_path):
        print(f'Downloading tflite file... (save path: {model_path})')
        urllib.request.urlretrieve(
            remote_path + os.path.basename(model_path),
            model_path,
            progress_print
        )
        print('\n')
    print('TFLite file is prepared!')
