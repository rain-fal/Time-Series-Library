# ================= 全局配置区域 (环境与数据) =================
# 1. 项目根目录
PROJECT_ROOT="D:/postgraduate_work/Time-Series-Library"

# 2. 定义要运行的模型列表 (注释掉不想运行的)
MODELS=(
  # "DLinear" 
  # "iTransformer" 
  # "PatchTST" 
  # "Autoformer" 
  # "Informer" 
  # "Transformer" 
  # "TimeMixer"
  # "Crossformer"
  "CrossLinear"
)

# 3. 定义预测长度列表
PRED_LENS=(96 192 336 720)

# 4. 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 5. 数据集通用参数
ROOT_PATH="./dataset/NW/风电/桂东"
DATA_PATH="桂东.csv"
DATA_TYPE="custom"
FEATURES="MSS"
SEQ_LEN=96
LABEL_LEN=48
ENC_IN=30
DEC_IN=30
C_OUT=30
TRAIN_EPOCHS=10
# ==========================================================

# 进入项目目录
cd "$PROJECT_ROOT" || { echo "❌ 错误: 无法进入目录 $PROJECT_ROOT"; exit 1; }

# 获取时间戳
batch_start_time=$(date "+%Y%m%d_%H%M%S" | tr -d '\r')
LOG_DIR="./logs/${batch_start_time}"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "🚀 开始批量实验 (独立命令模式)"
echo "📍 模型: ${MODELS[*]}"
echo "📍 长度: ${PRED_LENS[*]}"
echo "=========================================================="

for model_name in "${MODELS[@]}"
do
  for pred_len in "${PRED_LENS[@]}"
  do
    log_file="${LOG_DIR}/${model_name}_len${pred_len}.log"
    echo ""
    echo "▶️  [$(date "+%H:%M:%S")] 正在运行: $model_name | 长度: $pred_len"
    
    # ================= 针对每个模型完全独立的命令块 =================
    
    case "$model_name" in
      "DLinear")
        # DLinear: 线性模型，速度快，Batch Size 可大一点
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data_path "$DATA_PATH" \
          --model_id "Power_${SEQ_LEN}_${pred_len}" \
          --model "$model_name" \
          --data "$DATA_TYPE" \
          --features "$FEATURES" \
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in "$ENC_IN" \
          --dec_in "$DEC_IN" \
          --c_out "$C_OUT" \
          --d_model 512 \
          --d_ff 2048 \
          --batch_size 32 \
          --train_epochs "$TRAIN_EPOCHS" \
          --des 'Exp' \
          --itr 1 2>&1 | tee "$log_file"
        ;;
        
      "iTransformer")
        # iTransformer: 对 d_model 敏感
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data_path "$DATA_PATH" \
          --model_id "Power_${SEQ_LEN}_${pred_len}" \
          --model "$model_name" \
          --data "$DATA_TYPE" \
          --features "$FEATURES" \
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in "$ENC_IN" \
          --dec_in "$DEC_IN" \
          --c_out "$C_OUT" \
          --d_model 512 \
          --d_ff 512 \
          --n_heads 8 \
          --batch_size 16 \
          --train_epochs "$TRAIN_EPOCHS" \
          --des 'Exp' \
          --itr 1 2>&1 | tee "$log_file"
        ;;
        
      "PatchTST")
        # PatchTST: 显存占用较大，Batch Size 调小，有 patch_len 参数
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data_path "$DATA_PATH" \
          --model_id "Power_${SEQ_LEN}_${pred_len}" \
          --model "$model_name" \
          --data "$DATA_TYPE" \
          --features "$FEATURES" \
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --e_layers 1 \
          --d_layers 1 \
          --factor 3 \
          --enc_in "$ENC_IN" \
          --dec_in "$DEC_IN" \
          --c_out "$C_OUT" \
          --n_heads 2 \
          --train_epochs "$TRAIN_EPOCHS" \
          --des 'Exp' \
          --itr 1 2>&1 | tee "$log_file"
        ;;
        
      "TimeMixer")
        # TimeMixer: 包含降采样参数 down_sampling_layers
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data_path "$DATA_PATH" \
          --model_id "Power_${SEQ_LEN}_${pred_len}" \
          --model "$model_name" \
          --data "$DATA_TYPE" \
          --features "$FEATURES" \
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --e_layers 2 \
          --d_layers 1 \
          --enc_in "$ENC_IN" \
          --dec_in "$DEC_IN" \
          --c_out "$C_OUT" \
          --d_model 16 \
          --d_ff 32 \
          --down_sampling_layers 3 \
          --down_sampling_method avg \
          --down_sampling_window 2 \
          --learning_rate 0.0001 \
          --train_epochs "$TRAIN_EPOCHS" \
          --des 'Exp' \
          --itr 1 2>&1 | tee "$log_file"
        ;;

      "CrossLinear")
        # CrossLinear: 假设配置
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data_path "$DATA_PATH" \
          --model_id "Power_${SEQ_LEN}_${pred_len}" \
          --model "$model_name" \
          --data "$DATA_TYPE" \
          --features "$FEATURES" \
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --enc_in "$ENC_IN" \
          --dec_in "$DEC_IN" \
          --c_out "$C_OUT" \
          --patch_len 16 \
          --d_model 512 \
          --d_ff 1024 \
          --alpha 1 \
          --beta 0.5 \
          --train_epochs "$TRAIN_EPOCHS" \
          --des 'Exp' \
          --itr 1 2>&1 | tee "$log_file"
        ;;

      "Autoformer"|"Informer")
         # 其他模型通用配置 (如果启用的话)
         python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data_path "$DATA_PATH" \
          --model_id "Power_${SEQ_LEN}_${pred_len}" \
          --model "$model_name" \
          --data "$DATA_TYPE" \
          --features "$FEATURES" \
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in "$ENC_IN" \
          --dec_in "$DEC_IN" \
          --c_out "$C_OUT" \
          --train_epochs "$TRAIN_EPOCHS" \
          --des 'Exp' \
          --itr 1 2>&1 | tee "$log_file"
         ;;

        "Transformer"|"Crossformer")
         # 其他模型通用配置 (如果启用的话)
         python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data_path "$DATA_PATH" \
          --model_id "Power_${SEQ_LEN}_${pred_len}" \
          --model "$model_name" \
          --data "$DATA_TYPE" \
          --features "$FEATURES" \
          --seq_len "$SEQ_LEN" \
          --label_len "$LABEL_LEN" \
          --pred_len "$pred_len" \
          --e_layers 2 \
          --d_layers 1 \
          --enc_in "$ENC_IN" \
          --dec_in "$DEC_IN" \
          --c_out "$C_OUT" \
          --train_epochs "$TRAIN_EPOCHS" \
          --des 'Exp' \
          --itr 1 2>&1 | tee "$log_file"
         ;;
         
      *)
        echo "⚠️ 未定义的模型配置: $model_name"
        ;;
    esac

    # 检查状态
    if [ $? -eq 0 ]; then
        echo "✅ 成功: ${model_name} (len=${pred_len})"
    else
        echo "❌ 失败: ${model_name} (len=${pred_len}) 请检查日志 $log_file"
    fi
    
    sleep 3
  done
done

echo "🎉 完毕!"