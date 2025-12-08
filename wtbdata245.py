import pandas as pd


def process_wind_data(input_path, output_path, year=2023):
    """
    处理风电数据，按时间聚合Patv(OT)值

    参数：
        input_path: 输入文件路径
        output_path: 输出文件路径
        year: 指定年份（默认2023）
    """
    # 1. 读取数据
    df = pd.read_csv(input_path)

    # 2. 构造完整时间戳（处理跨年情况）
    base_date = pd.Timestamp(f"{year}-01-01")
    df['datetime'] = base_date + pd.to_timedelta(df['Day'] - 1, unit='D') + pd.to_timedelta(df['Tmstamp'] + ':00')

    # 3. 处理Patv列
    df['Patv'] = pd.to_numeric(df['Patv'], errors='coerce').fillna(0)  # 非数值转为0

    # 4. 按时间聚合
    result = (
        df.groupby('datetime', as_index=False)
        .agg(OT=('Patv', 'sum'))
        .sort_values('datetime')
    )

    # 5. 保存结果
    result.to_csv(output_path, index=False)
    print(f"处理完成，结果已保存到 {output_path}")
    print("示例数据：")
    return result.head()


# 使用示例
process_wind_data('dataset/Power/wtbdata_245days.csv', 'SDWPF.csv')