import argparse
import json
import logging
import math
import os
import random
import numpy as np
from dataclasses import dataclass
from datasets import DatasetDict
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="tmp/multi_train",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to the cache.",
        default="./cache/"
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test.json", help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--context_file", type=str, default="context.json", help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--output_file", type=str, default="./result.json", help="Where to store the final model."
        )

    args = parser.parse_args()
    return args

@dataclass
class DataCollatorForMultipleChoice:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __call__(self, features):
        #label_name = "relevant" if "relevant" in features[0].keys() else "labels"
        #labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        #batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def main():
    args=parse_args()
    test_data=DatasetDict.from_json({'test':args.test_file})
    with open(args.context_file,encoding="utf-8") as f:
        context=json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        cache_dir=args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))

    index_list=[0,1,2,3]
    def preprocess_function(examples):
        #print('======Example======: ',examples['paragraphs'][0][0])
        first_sentences = [[context] * 4 for context in examples["question"]]
        question_headers = examples['question']
        second_sentences = [
            [f"{context[examples['paragraphs'][i][index]]}" for index in index_list] for i,question in enumerate(question_headers)
        ]
        #labels = examples[label_column_name]
       
        

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=512,
            padding=True,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        #tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
    
    model_checkpoint="bert-base-chinese"
    model_name = model_checkpoint.split("/")[-1]
    batch_size=2
    train_args = TrainingArguments(
        output_dir=f"./{model_name}-finetuned",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )
    trainer = Trainer(
        model,
        train_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        
    )

    encoded_datasets=test_data.map(preprocess_function, batched=True)
    results=trainer.predict(encoded_datasets['test'])

    pred=np.argmax(results.predictions,axis=1)
    test_data['test'] = test_data['test'].add_column("label",pred)

    with open(args.output_file, "w") as outfile:
        dic_list=[]
        for i in range(len(pred)):
            diction={
                'id':test_data['test']['id'][i],
                'question':test_data['test']['question'][i],
                'paragraphs':test_data['test']['paragraphs'][i],
                'label':test_data['test']['label'][i],
            }
            dic_list.append(diction)
        json_object = json.dumps(dic_list, indent=4,ensure_ascii=False)
        outfile.write(json_object)


if __name__ == "__main__":
    main()