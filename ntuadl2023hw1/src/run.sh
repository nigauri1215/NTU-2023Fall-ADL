accelerate launch multi_test.py \
  --model_name_or_path multi_train_wwm \
  --test_file ${2} \
  --context_file ${1} \
  --output_file ./result.json \
  --cache_dir ./cache/ \
  --max_seq_length 512

accelerate launch QA_test.py \
  --model_name_or_path qa_train_wwm \
  --test_file ./result.json \
  --context_file ${1} \
  --output_file ${3} \
  --cache_dir ./cache/ \
  --max_seq_length 512 \
  --n_best_size 20