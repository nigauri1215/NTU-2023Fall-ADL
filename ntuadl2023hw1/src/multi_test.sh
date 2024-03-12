accelerate launch multi_test.py \
  --model_name_or_path tmp/multi_train_wwm \
  --test_file data/test.json \
  --context_file data/context.json \
  --output_file ./result.json \
  --cache_dir ./cache/ \
  --max_seq_length 512