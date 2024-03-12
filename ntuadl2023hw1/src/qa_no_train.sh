# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#accelerate launch run_swag_no_trainer.py \
#  --model_name_or_path bert-base-uncased \
#  --dataset_name swag \
#  --output_dir /tmp/test-swag-no-trainer \
#  --pad_to_max_length

#--resume_from_checkpoint tmp/qa_train_wwm/step_10000 \

accelerate launch adlhw1_qa_no_train.py \
  --config_name bert-base-chinese \
  --tokenizer_name bert-base-chinese \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file  data/context.json \
  --cache_dir ./cache/ \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --max_seq_length 512 \
  --n_best_size 20 \
  --max_answer_length 60 \
  --output_dir tmp/qa_no_train/

