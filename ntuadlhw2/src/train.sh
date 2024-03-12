accelerate launch train.py \
  --model_name_or_path google/mt5-small \
  --output_dir tmp/train_1209 \
  --train_file data/train.jsonl \
  --text_column maintext \
  --summary_column title \
  --max_source_length 256 \
  --max_target_length 64 \
  --pad_to_max_length \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_beams 10 \
  #--do_sample True \
  #--top_p 0.5 \
  #--top_k 10 \
  #--num_beams 1 \
  #--temperature 0.7 \