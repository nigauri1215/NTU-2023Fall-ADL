accelerate launch test.py \
  --model_name_or_path tmp/temp_07 \
  --output_dir result/test_1209_temp_2.jsonl \
  --test_file data/public.jsonl \
  --text_column maintext \
  --max_source_length 256 \
  --max_target_length 64 \
  --num_beams 15 \
