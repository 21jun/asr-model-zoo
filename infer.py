import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf

model = Speech2TextForConditionalGeneration.from_pretrained(
    "facebook/s2t-medium-librispeech-asr"
)
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-librispeech-asr")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    print(batch)
    return batch


def from_file(file):
    speech, _ = sf.read(file)
    return speech


# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# ds = ds.map(map_to_array)

# input_features = processor(
#     ds["speech"][0], sampling_rate=16_000, return_tensors="pt"
# ).input_features  # Batch size 1

filepath = "sample/84-121123-0001.wav"
input_features = processor(
    from_file(filepath), sampling_rate=16_000, return_tensors="pt"
).input_features  # Batch size 1

# https://huggingface.co/transformers/main_classes/model.html
generated_ids = model.generate(
    input_ids=input_features, num_beams=10, num_return_sequences=10
)
print(generated_ids)
transcription = processor.batch_decode(generated_ids)

for t in transcription:
    print(t)


# model.save_pretrained("model")
# processor.save_pretrained("processor")
