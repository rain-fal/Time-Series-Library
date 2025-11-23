export CUDA_VISIBLE_DEVICES=0

model_name=Informer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Power \
  --data_path 东起风电实际功率.csv \
  --model_id Power_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Power \
  --data_path 东起风电实际功率.csv \
  --model_id Power_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Power \
  --data_path 东起风电实际功率.csv \
  --model_id Power_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Power \
  --data_path 东起风电实际功率.csv \
  --model_id Power_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1