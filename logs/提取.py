import os
import re
import pandas as pd

def parse_logs_to_matrix(root_path):
    data_list = []
    
    # 1. 定义正则匹配模式
    # 匹配日志中的 mse 和 mae 结果
    pattern_metrics = re.compile(r'mse:(\d+\.\d+),\s*mae:(\d+\.\d+)')
    # 匹配日志开头的模型名称 (例如 Model: PatchTST)
    pattern_model = re.compile(r'Model:\s+(\w+)')
    # 匹配预测长度 (优先从 Pred Len 提取)
    pattern_pred_len = re.compile(r'Pred Len:\s+(\d+)')

    # 2. 遍历文件提取数据
    print(f"开始遍历文件夹: {root_path} ...")
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # A. 提取测试指标 (MSE/MAE)
                        # 我们取最后一次出现的 mse/mae (通常是 Test set 的结果)
                        metrics_match = list(pattern_metrics.finditer(content))
                        if not metrics_match:
                            continue
                        final_metric = metrics_match[-1] # 取最后一行
                        mse = float(final_metric.group(1))
                        mae = float(final_metric.group(2))

                        # B. 提取模型名称 (Model Name)
                        model_match = pattern_model.search(content)
                        # 如果日志里没写，尝试从文件名猜 (例如 PatchTST_...)
                        if model_match:
                            model_name = model_match.group(1)
                        else:
                            model_name = file.split('_')[0] 

                        # C. 提取预测步长 (Seq Len / Pred Len)
                        # 优先从内容找 'Pred Len'
                        len_match = pattern_pred_len.search(content)
                        if len_match:
                            pred_len = int(len_match.group(1))
                        else:
                            # 找不到则尝试从文件名找 len192
                            len_file_match = re.search(r'len(\d+)', file)
                            pred_len = int(len_file_match.group(1)) if len_file_match else 0

                        # D. 存入列表
                        data_list.append({
                            'Model': model_name,
                            'Len': pred_len,
                            'MSE': mse,
                            'MAE': mae
                        })
                except Exception as e:
                    print(f"跳过文件 {file}: {e}")

    # 3. 生成 DataFrame 并重塑为目标表格格式
    if not data_list:
        return pd.DataFrame()

    df = pd.DataFrame(data_list)

    # 这里的 pivot 逻辑：
    # Index (行) = Len (预测步长)
    # Columns (列) = Model (模型名称)
    # Values (值) = MSE, MAE
    pivot_df = df.pivot_table(index='Len', columns='Model', values=['MSE', 'MAE'])

    # 4. 调整列的层级顺序，使其变成 [Model -> MSE/MAE] 的格式 (这也是截图中的格式)
    # 目前的列索引是 (Metric, Model)，我们需要 swaplevel 变成 (Model, Metric)
    pivot_df = pivot_df.swaplevel(0, 1, axis=1)
    
    # 对列名进行排序，让同一个 Model 的 MSE 和 MAE 挨在一起
    pivot_df = pivot_df.sort_index(axis=1)

    return pivot_df

# --- 执行脚本 ---
folder_path = './20251204_102215'  # 修改为您的日志文件夹路径
result_matrix = parse_logs_to_matrix(folder_path)

if not result_matrix.empty:
    print("\n生成的对比表格：")
    print(result_matrix)
    
    # 导出到 CSV
    output_file = 'Model_Compare_Result.csv'
    result_matrix.to_csv(output_file) # CSV 不需要 openpyxl
    print(f"\n已保存为 CSV 文件: {output_file}")
else:
    print("未提取到有效数据，请检查日志格式。")