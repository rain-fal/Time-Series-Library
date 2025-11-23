# 1. 强制进入项目根目录
cd D:/postgraduate_work/Time-Series-Library

# 2. 创建日志文件夹
if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

set CUDA_VISIBLE_DEVICES=0
model_name=PatchTST

for pred_len in 96 192 336 720
do
  # 生成带时间戳的日志文件名
  current_time=$(date "+%Y%m%d_%H%M%S")
  log_file="logs/${model_name}_${current_time}_桂东_${pred_len}.log"

  echo "----------------------------------------------------------"
  echo "正在运行: $model_name, 预测长度: $pred_len"
  echo "日志将保存至: $log_file"
  echo "----------------------------------------------------------"

  # =======================================================
  # 核心修改：使用 '| tee' 替代 '>'
  # 1. 2>&1 : 把错误信息也汇入标准输出
  # 2. | tee "$log_file" : 管道传给 tee，tee 会负责"一分二"，一份给屏幕，一份给文件
  # =======================================================
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/NW/风电/桂东 \
    --data_path 桂东.csv \
    --model_id Power_96'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 30 \
  --dec_in 30 \
  --c_out 30 \
  --des 'Exp' \
  --n_heads 2 \
    --itr 1 2>&1 | tee "$log_file"

  echo "完成!"
done