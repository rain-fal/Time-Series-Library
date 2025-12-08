export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=5
for pred_len in 96 192 336 720
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path  ./dataset/nw\
    --data_path 2023-2025_桂东_六林冲风电场_468239767043_玉林市_北流_风电.csv \
    --model_id SDWPF'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 22 \
    --dec_in 22 \
    --c_out 22 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 1 \
    --top_k 5
done