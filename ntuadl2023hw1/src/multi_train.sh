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

# hfl/chinese-roberta-wwm-ext 

accelerate launch adlhw1_multi.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --train_file data/train.json \
  --validation_file data/valid.json \
  --context_file data/context.json \
  --cache_dir ./cache/ \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 3 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --output_dir tmp/multi_train_wwm/