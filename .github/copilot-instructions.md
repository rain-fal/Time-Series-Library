# Time-Series-Library 项目 AI 编码助手指南

## 概述
Time-Series-Library (TSLib) 是一个开源库，专为深度时间序列分析设计。它支持五大主要任务：长短期预测、数据插补、异常检测和分类。该库的结构旨在便于评估先进模型和开发新模型。

## 关键组件
- **模型**：位于 `./models` 目录，该目录包含各种时间序列模型的实现。添加新模型时，请参考 `Transformer.py` 的结构。
- **实验脚本**：存放于 `./scripts`，这些脚本用于训练和评估模型。脚本按任务和数据集进行组织。
- **数据集**：将预处理后的数据集放置在 `./dataset` 文件夹中。支持的数据集可从 Google Drive、百度网盘或 Hugging Face 下载。
- **实验框架**：核心实验逻辑位于 `./exp` 中。在 `exp_basic.py` 的 `Exp_Basic.model_dict` 中注册新模型。

## 开发者工作流
### 环境设置
1. 安装 Python 3.8。
2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
3. 下载数据集并放置到 `./dataset` 中。

### 运行实验
使用提供的脚本复现实验结果：
```bash
# 长期预测
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh

# 短期预测
bash ./scripts/short_term_forecast/TimesNet_M4.sh

# 数据插补
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh

# 异常检测
bash ./scripts/anomaly_detection/PSM/TimesNet.sh

# 分类
bash ./scripts/classification/TimesNet.sh
```

### 添加新模型
1. 将模型文件添加到 `./models`。
2. 在 `./exp/exp_basic.py` 的 `Exp_Basic.model_dict` 中注册模型。
3. 在 `./scripts` 中创建对应的实验脚本。

### 调试
- 使用 `logs/` 目录查看实验日志。
- 检查点保存在 `./checkpoints` 中，可用于模型恢复。

## 项目特定约定
- **模型结构**：遵循 `Transformer.py` 中的模块化设计模式。
- **脚本**：确保脚本是任务特定的，并放置在 `./scripts` 下的相应子目录中。
- **数据集处理**：使用 `data_provider` 模块以确保数据加载的一致性。

## 外部依赖
- **PyTorch**：核心深度学习框架。
- **预处理数据集**：可从 `README.md` 中链接的公共资源库获取。

## AI 助手提示
- 参考 `README.md` 获取任务和模型的详细描述。
- 遵循 `CONTRIBUTING.md` 中的贡献指南，添加新功能或修复错误。
- 使用 `./test_results` 目录验证新实现。

## 联系方式
如有问题或建议，请参考 `README.md` 中的“联系方式”部分，或在仓库中提交 issue。

---
本文档旨在帮助 AI 编码助手高效理解并贡献 Time-Series-Library 项目。如需进一步澄清，请咨询项目维护者。