import tensorflow as tf

#from datasets import load_dataset
from transformers import WhisperProcessor, WhisperFeatureExtractor, TFWhisperForConditionalGeneration, WhisperTokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en", predict_timestamps=True)
processor = WhisperProcessor(feature_extractor, tokenizer)
model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
# Loading dataset
#ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

import librosa
audio, sr = librosa.load("demo.wav", sr=16000)
inputs = feature_extractor(
    audio, sampling_rate=sr, return_tensors="tf"
)
input_features = inputs.input_features

# %%
# loaded model... now with generate!
tflite_model_path = 'whisper-tiny-en.tflite'
interpreter = tf.lite.Interpreter(tflite_model_path)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.set_tensor(input_details[0]['index'], input_features)
interpreter.invoke()

generated_ids = interpreter.get_tensor(output_details[0]['index'])

#tflite_generate = interpreter.get_signature_runner()
#generated_ids = tflite_generate(input_features=input_features)["sequences"]
print(generated_ids)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)

