import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=True,
        help="do sample for summary or not",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="maximum value of ",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.test_file is None:
        raise ValueError("Need either a dataset name or a training/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json","jsonl"], "`train_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json","jsonl"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def main():
    args = parse_args()

    send_example_telemetry("run_summarization_no_trainer", args)

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
        if extension == "jsonl":
            extension = "json"

        raw_datasets = load_dataset(extension, data_files=data_files)


    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    column_names = raw_datasets["test"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False


    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        text_id=examples['id']
        text_id=[int(id) for id in text_id]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        model_inputs['id']=text_id
        return model_inputs

    with accelerator.main_process_first():
        test_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        return preds

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model,test_dataloader
    )

    model.eval()

    gen_kwargs = {
            "max_length": args.val_max_target_length,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        }
    
    predictions=[]
    id_list=[]
    for step, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            print("batch: ",batch)
            pred_id=batch["id"]

            generated_tokens = generated_tokens.cpu().numpy()
            pred_id=pred_id.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_preds = postprocess_text(decoded_preds)
            for i, result in enumerate(decoded_preds):
                predictions.append(result)
                cur_id=pred_id[i].tolist()
                id_list.append(cur_id)
    
    with open(args.output_dir,"w",encoding="utf8") as f:
        for i in range(len(predictions)):
            json.dump({"title":predictions[i],"id":str(id_list[i])}, f,ensure_ascii = False)
            
            f.write("\n")

if __name__ == "__main__":
    main()