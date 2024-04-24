#!pip install transformers
#!pip install datasets

# %%
import tensorflow as tf

from datasets import load_dataset
from transformers import WhisperProcessor, WhisperFeatureExtractor, TFWhisperForConditionalGeneration, WhisperTokenizer

generate_saved_model = False
generate_tflite_model = False

quantize = True

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en", predict_timestamps=True)
processor = WhisperProcessor(feature_extractor, tokenizer)
model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
# Loading dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = feature_extractor(
    ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="tf"
)
input_features = inputs.input_features

# Generating Transcription
if generate_saved_model:
  generated_ids = model.generate(input_features=input_features)
  print(generated_ids)
  transcription = processor.tokenizer.decode(generated_ids[0])
  print(transcription)
  model.save('./content/tf_whisper_saved')

# %% [markdown]
# ##Convert saved model to TFLite model

# %%
import tensorflow as tf

saved_model_dir = './content/tf_whisper_saved'
tflite_model_path = 'whisper.tflite'

# Convert the model
if generate_saved_model:
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  with open(tflite_model_path, 'wb') as f:
      f.write(tflite_model)

# %% [markdown]
# ## Create generation-enabled TF Lite model
# 
# The solution consists in defining a model whose serving function is the generation call. Here's an example of how to do it:

# %%
class GenerateModel(tf.Module):
  def __init__(self, model):
    super(GenerateModel, self).__init__()
    self.model = model

  @tf.function(
    # shouldn't need static batch size, but throws exception without it (needs to be fixed)
    input_signature=[
      tf.TensorSpec((1, 80, 3000), tf.float32, name="input_features"),
    ],
  )
  def serving(self, input_features):
    outputs = self.model.generate(
      input_features,
      max_new_tokens=450, #change as needed
      return_dict_in_generate=True,
    )
    return {"sequences": outputs["sequences"]}

saved_model_dir = './content/tf_whisper_saved'

if generate_saved_model:
  generate_model = GenerateModel(model=model)
  tf.saved_model.save(generate_model, saved_model_dir, signatures={"serving_default": generate_model.serving})

def representative_dataset():
    num_datasets = 1 # max 73
    for i in range(num_datasets):#Change this to 100 and provide 100 different audio files from known dataset 
      inputs = feature_extractor(
          ds[i]["audio"]["array"], sampling_rate=ds[i]["audio"]["sampling_rate"], return_tensors="tf"
      )
      input_features = inputs.input_features
      attention = tf.constant(0, shape=(1, 1), dtype=tf.int32)
      input_ids = tf.constant(0, shape=(1, 1), dtype=tf.int32)
      yield [attention, input_ids, input_features]

# Convert the model
if generate_tflite_model:
  if not quantize:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
  else:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32 # int32 can not selected for int8
    tflite_model = converter.convert()

# Save the model
if quantize:
  tflite_model_path = 'whisper-tiny-en-int8.tflite'
else:
  tflite_model_path = 'whisper-tiny-en-float.tflite'

if generate_tflite_model:
  with open(tflite_model_path, 'wb') as f:
      f.write(tflite_model)

# %%
# loaded model... now with generate!  
interpreter = tf.lite.Interpreter(tflite_model_path)

if quantize:
  interpreter.allocate_tensors()
  for i in range(450):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    attention = tf.constant(0, shape=(1, 1), dtype=tf.int32)
    input_ids = tf.constant(0, shape=(1, 1), dtype=tf.int32)
    interpreter.set_tensor(input_details[0]['index'], attention)
    interpreter.set_tensor(input_details[1]['index'], input_ids)
    interpreter.set_tensor(input_details[2]['index'], input_features)
    interpreter.invoke()
    generated_ids = interpreter.get_tensor(output_details[0]['index'])
    print(generated_ids)
else:
  tflite_generate = interpreter.get_signature_runner()
  generated_ids = tflite_generate(input_features=input_features)["sequences"]
  print(generated_ids)
  transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  print(transcription)


