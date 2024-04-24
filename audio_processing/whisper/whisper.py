import enum
import sys
import time

import librosa


# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results, write_predictions  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# Parameters 1
# ======================
INPUT_PATH = 'demo.wav'


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Whisper Speech To Text', INPUT_PATH, None
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite


# ======================
# Parameters 2
# ======================
MODEL_PATH = f'whisper-tiny-en.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/whisper/'


# ======================
# Main functions
# ======================
def recognize_from_audio():
    #from datasets import load_dataset
    from transformers import WhisperProcessor, WhisperFeatureExtractor, TFWhisperForConditionalGeneration, WhisperTokenizer

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en", predict_timestamps=True)
    processor = WhisperProcessor(feature_extractor, tokenizer)
    model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    # Loading dataset
    #ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    audio, sr = librosa.load("demo.wav", sr=16000)
    inputs = feature_extractor(
        audio, sampling_rate=sr, return_tensors="tf"
    )
    input_features = inputs.input_features

    # %%
    # loaded model... now with generate!
    tflite_model_path = 'whisper-tiny-en.tflite'
    if args.tflite:
        interpreter = tf.lite.Interpreter(tflite_model_path)
    else:
        interpreter = ailia_tflite.Interpreter(tflite_model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)

    if args.benchmark:
        print('BENCHMARK mode')
        average_time = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            interpreter.set_tensor(input_details[0]['index'], input_features)
            interpreter.invoke()
            end = int(round(time.time() * 1000))
            average_time = average_time + (end - start)
            print(f'\tailia processing time {end - start} ms')
        print(f'\taverage time {average_time / args.benchmark_count} ms')
    else:
        interpreter.set_tensor(input_details[0]['index'], input_features)
        interpreter.invoke()

    generated_ids = interpreter.get_tensor(output_details[0]['index'])

    print(generated_ids)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)

    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    recognize_from_audio()


if __name__ == '__main__':
    main()







