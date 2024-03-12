accelerate launch QA_test.py \
  --model_name_or_path tmp/qa_train_wwm \
  --test_file ./result.json \
  --context_file data/context.json \
  --output_file ./pred.csv \
  --cache_dir ./cache/ \
  --max_seq_length 512 \
  --n_best_size 20