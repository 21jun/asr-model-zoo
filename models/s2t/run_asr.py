import soundfile as sf

from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

from transformers import Trainer, TrainingArguments

from torch.utils.data import Dataset
import librosa
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import torch


model = Speech2TextForConditionalGeneration.from_pretrained(
    "facebook/s2t-small-librispeech-asr"
)
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


class LibriSpeechDataset(Dataset):
    def __init__(self, json_path, processor):
        self.json_path = json_path
        self.data = self.load_data_from_json(json_path)
        self.processor = processor

    def load_data_from_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        data = data["data"][100:]
        return data

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.data[idx]["file"], 16000)
        input_value = self.processor.feature_extractor(audio, sampling_rate=16000)
        # Do some text preprocessing here
        text = self.data[idx]["text"]
        with self.processor.as_target_processor():
            label = self.processor(text).input_ids

        # print(input_value)
        sample = {"input_values": input_value["input_features"][0], "labels": label}
        return sample

    def __len__(self):
        return len(self.data)


train_dataset = LibriSpeechDataset(
    "/root/develop/KIWI-module/code/wav2byte-pipeline/data/en-librispeech-test-clean-pure-99.0-local-wav.json",
    processor,
)


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Speech2TextProcessor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_features": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # lang_features = [{"lang": feature['lang'] for feature in features}]

        # print("@ @ start")
        # feature_extractor 를 명시해준것은, 현재 processor 구현에 pad를 매칭이 안됨 (feature_extractor 에는 있음)
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # print("$ $ End")

        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(  # tokenizer를 명시해준것은, 현재 processor 구현에 pad를 매칭이 안됨 (tokenizer에는 있음)
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        # print(batch)
        return batch


data_collator = DataCollatorWithPadding(processor=processor, padding=True)
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
trainer.train()
