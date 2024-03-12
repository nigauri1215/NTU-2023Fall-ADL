import argparse
import json
import logging
import collections
import math
import os
import random
from pathlib import Path
import pandas as pd

import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions
from datasets import DatasetDict
from typing import Optional, Tuple
import numpy as np

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="tmp/qa_train",
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
        "--test_file", type=str, default="result.json", help="A csv or a json file containing the training data."
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
        "--n_best_size",
        type=int,
        help="the maximum length for feature.",
        default=20
    )
    parser.add_argument(
        "--output_file", type=str, default="./pred.csv", help="Where to store the final model."
        )

    args = parser.parse_args()
    return args

def main():
    args=parse_args()
    model_checkpoint=args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint,cache_dir=args.cache_dir,)
    test_data=DatasetDict.from_json({'test':args.test_file})
    with open(args.context_file,encoding="utf-8") as f:
        context=json.load(f)

    context_list=[]
    for k in range(len(test_data['test'])):
        idx=test_data['test'][k]['label']
        tmp=context[test_data['test'][k]['paragraphs'][idx]]
        context_list.append(tmp)

    test_data['test']=test_data['test'].add_column("context",context_list)

    max_length = args.max_seq_length
    doc_stride = 128
    pad_on_right = tokenizer.padding_side == "right"

    model_checkpoint="bert-base-chinese-qa"
    model_name = model_checkpoint.split("/")[-1]

    train_args = TrainingArguments(
        f"{model_name}-finetuned-squad",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        push_to_hub=False,
        
    )
    
    data_collator = default_data_collator

    trainer = Trainer(
        model,
        train_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    validation_features = test_data["test"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=test_data["test"].column_names
    )
    raw_predictions = trainer.predict(validation_features)

    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    max_answer_length = 50


    logger = logging.getLogger(__name__)

    def postprocess_qa_predictions(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        if len(predictions) != 2:
            raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                offset_mapping = features[feature_index]["offset_mapping"]
                token_is_max_context = features[feature_index].get("token_is_max_context", None)

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(
                output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
            )
            nbest_file = os.path.join(
                output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        return all_predictions
    
    final_predictions = postprocess_qa_predictions(test_data["test"], validation_features, raw_predictions.predictions)
    v_list=[]
    k_list=[]
    for k,v in final_predictions.items():
        v_list.append(v)
        k_list.append(k)
        frame=pd.DataFrame({'id':k_list,'answer':v_list})
        frame
        frame.to_csv(args.output_file,index=False,encoding='utf_8_sig')


if __name__ == "__main__":
    main()

