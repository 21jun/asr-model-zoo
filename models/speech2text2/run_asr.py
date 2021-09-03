from transformers import (
    SpeechEncoderDecoderModel,
    Speech2Text2Processor,
    Speech2TextProcessor,
)
import soundfile as sf

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from transformers import Trainer, TrainingArguments

from torch.utils.data import Dataset
import librosa
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

import torch
from torch import nn


from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    is_apex_available,
)

parser = HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()[0]


processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
# model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
model = SpeechEncoderDecoderModel.from_pretrained("saved_model")


# model.save_pretrained("saved_model")

model.config.gradient_checkpointing = True

model.encoder.feature_extractor._freeze_parameters()

s2t_processor = Speech2TextProcessor.from_pretrained(
    "facebook/s2t-medium-librispeech-asr"
)


class LibriSpeechDataset(Dataset):
    def __init__(self, json_path, tokenizer, feature_extractor):
        self.json_path = json_path
        self.data = self.load_data_from_json(json_path)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def load_data_from_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        data = data["data"][100:]
        return data

    def __getitem__(self, idx):
        audio, _ = librosa.load(self.data[idx]["file"], 16000)
        input_value = self.feature_extractor.feature_extractor(
            audio, sampling_rate=16000
        )
        # Do some text preprocessing here
        text = self.data[idx]["text"]
        with self.tokenizer.as_target_processor():
            label = self.tokenizer(text).input_ids

        # print(input_value)
        sample = {"input_values": input_value["input_values"][0], "labels": label}
        return sample

    def __len__(self):
        return len(self.data)


train_dataset = LibriSpeechDataset(
    "/root/develop/KIWI-module/code/wav2byte-pipeline/data/en-librispeech-test-clean-pure-99.0-local-wav.json",
    tokenizer=s2t_processor,
    feature_extractor=processor,
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
    feature_extractor: Speech2Text2Processor
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
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # lang_features = [{"lang": feature['lang'] for feature in features}]

        print("@ @ start")
        # feature_extractor 를 명시해준것은, 현재 processor 구현에 pad를 매칭이 안됨 (feature_extractor 에는 있음)
        batch = self.feature_extractor.feature_extractor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        print("$ $ End")

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

        return batch


class MyTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)
        input_ids = inputs["labels"]
        print("INPUT======")
        print(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        print("====OUTPUT====")
        print(outputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


data_collator = DataCollatorWithPadding(
    processor=s2t_processor, feature_extractor=processor, padding=True
)


trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=s2t_processor.feature_extractor,
)

trainer.train()
