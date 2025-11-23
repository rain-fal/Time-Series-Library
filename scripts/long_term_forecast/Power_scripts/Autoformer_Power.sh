export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

for pred_len in  96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path  ./dataset/Power \
    --data_path 东起风电实际功率.csv \
    --model_id Power_96'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --itr 1
done
