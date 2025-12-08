import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
pred1 = np.load('../results/long_term_forecast_Power_96_96_Autoformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/pred.npy')
pred2 = np.load('../results/long_term_forecast_风电_96_96_SSCNN_custom_ftS_sl96_ll48_pl96_dm8_nh8_el4_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/pred.npy')
true = np.load('../results/long_term_forecast_Power_96_96_Autoformer_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/true.npy')

print(f"pred1.shape = {pred1.shape}, pred2.shape = {pred2.shape}, true.shape = {true.shape}")
# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'  # SimHei 是黑体，可以根据系统调整为其他中文字体

plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常
# 2. 样本索引
sample_idx = 1600

# 3. 提取历史与未来
history = true[sample_idx-1, :, 0]   # 历史真实
true_fut = true[sample_idx, :, 0]    # 未来真实
pred1_fut = pred1[sample_idx, :, 0]
pred2_fut = pred2[sample_idx, :, 0]

H = len(history)
F = len(true_fut)

# 4. 构造横坐标
x_hist = np.arange(H)
x_fut  = np.arange(H, H+F)

# 5. 为预测曲线添加历史连接点
#    在 x=H-1 处，预测线从 history[-1] 开始
x_pred_full = np.concatenate([[H-1], x_fut])
y_pred1_full = np.concatenate([[history[-1]], pred1_fut])
y_pred2_full = np.concatenate([[history[-1]], pred2_fut])

# 6. 画图
plt.figure(figsize=(8,8), dpi=150)

# 历史+未来真实值（一条连续线）
plt.plot(
    np.concatenate([x_hist, x_fut]),
    np.concatenate([history, true_fut]),
    label='真实值',
    color='#1f77b4',
    linewidth=2
)

# 两条预测线，都从历史末点接入
plt.plot(x_pred_full, y_pred1_full, label='NS-GNNCrossformer', color='#ff7f0e', linewidth=2)
plt.plot(x_pred_full, y_pred2_full, label='Copula', color='#2ca02c', linewidth=2)

plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, H+F-1)
plt.tight_layout()
plt.show()