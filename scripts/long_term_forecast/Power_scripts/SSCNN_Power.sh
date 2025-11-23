export CUDA_VISIBLE_DEVICES=0

model_name=SSCNN

for pred_len in  96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Power \
    --data_path 东起风电实际功率.csv \
    --model_id 风电_96_$pred_len \
    --model $model_name \
    --data custom \
    --features S \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --cycle_len 96 \
    --short_period_len 8 \
    --kernel_size 2 \
    --e_layers 4 \
    --d_layers 1 \
    --spatial 0 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --d_model 8 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --itr 1 \
    --train_epoch 200 \
    --patience 5
done