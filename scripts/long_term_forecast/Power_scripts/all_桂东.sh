
# ================= 全局配置区域 =================
# 1. 项目根目录
PROJECT_ROOT="D:/postgraduate_work/Time-Series-Library"

# 2. 定义要运行的模型列表
MODELS=( "DLinear" )
# "DLinear" "iTransformer" "PatchTST"
# "Autoformer" "Informer"  "Transformer"
# 3. 定义预测长度列表
PRED_LENS=(96 192 336 720)

# 4. 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 5. 数据集通用参数 (通常不随模型改变)
ROOT_PATH="./dataset/NW/风电/桂东"
DATA_PATH="桂东.csv"
DATA_TYPE="custom"
FEATURES="MSS"  # M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量
SEQ_LEN=96
LABEL_LEN=48
ENC_IN=30
DEC_IN=30
C_OUT=30
TRAIN_EPOCHS=10
batch_size=32
factor=1
d_model=512
d_ff=2048
n_heads=8
# ===============================================

# 进入项目目录
cd D:/postgraduate_work/Time-Series-Library || { echo "❌ 错误: 无法进入目录 $PROJECT_ROOT"; exit 1; }

# 2. 获取时间戳 (增加 tr -d '\r' 去除 Windows 可能产生的回车符)
batch_start_time=$(date "+%Y%m%d_%H%M%S" | tr -d '\r')

# 3. 定义并创建日志目录
LOG_DIR="./logs/${batch_start_time}"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "🚀 开始批量实验自动化脚本"
echo "📍 包含模型: ${MODELS[*]}"
echo "📍 包含长度: ${PRED_LENS[*]}"
echo "=========================================================="

# --- 外层循环：遍历模型 ---
for model_name in "${MODELS[@]}"
do
  # --- 内层循环：遍历预测长度 ---
  for pred_len in "${PRED_LENS[@]}"
  do
    
    # ================= [关键修改] 模型独立参数配置区 =================
    # 在这里为每个模型单独设置参数 (如层数, batch size, d_model等)
    # 如果没列出的模型，会使用 *) 中的默认参数
    
    case "$model_name" in
      "DLinear")
        # --- DLinear 专用配置 ---
        e_layers=2
        d_layers=1
        factor=3
        ;;
        
      "iTransformer")
        # --- iTransformer 专用配置 (参数通常较大) ---
        e_layers=2   # iTransformer 官方推荐有时是 2 或 3
        d_layers=1
        d_model=128
        d_ff=128
        factor=3
        ;;
        
      "PatchTST")
        # --- PatchTST 专用配置 ---
        e_layers=1
        d_layers=1
        factor=3
        n_heads=2
        ;;
        
      "Autoformer")
        # --- 其他未指定模型的默认配置 ---
        e_layers=2
        d_layers=1
        factor=3
        ;;
        
      "Informer")
        # --- 其他未指定模型的默认配置 ---
        e_layers=2
        d_layers=1
        factor=3
        ;;

      "Transformer")
        # --- 其他未指定模型的默认配置 ---
        e_layers=2
        d_layers=1
        ;;  
    esac
    # ================================================================

    # 文件名格式: 模型名_长度.log (因为文件夹已经是时间戳了，文件名里可以不再加时间，保持简洁)
    log_file="${LOG_DIR}/${model_name}_len${pred_len}.log"

    echo ""
    echo "▶️  [$(date "+%H:%M:%S")] 正在运行: $model_name | 长度: $pred_len"
    echo "    📄 日志: $log_file"

    # ================= 运行命令 =================
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
      --e_layers "$e_layers" \
      --d_layers "$d_layers" \
      --factor "$factor" \
      --enc_in "$ENC_IN" \
      --dec_in "$DEC_IN" \
      --c_out "$C_OUT" \
      --d_model "$d_model" \
      --d_ff "$d_ff" \
      --batch_size "$batch_size" \
      --train_epochs "$TRAIN_EPOCHS" \
      --des 'Exp' \
      --n_heads "$n_heads" \
      --itr 1 2>&1 | tee "$log_file"

    # 检查状态
    if [ $? -eq 0 ]; then
        echo "✅ 成功: ${model_name} (len=${pred_len})"
    else
        echo "❌ 失败: ${model_name} (len=${pred_len}) 请检查日志!"
    fi
    
    # 显存回收缓冲
    sleep 5

  done
done

echo "=========================================================="
echo "🎉 所有实验运行完毕!"
echo "=========================================================="