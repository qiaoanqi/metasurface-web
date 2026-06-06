# AI 超表面结构色设计系统

> 长沙理工大学 物电学院 光电2501  
> 乔安琪、谢家珞、侯琢  
> 《人工智能基础B》期末大作业

## 项目简介

基于深度学习和物理模型的超表面纳米柱结构色智能设计平台。支持 TiO₂、a-Si 等多种材料的单柱/双柱纳米结构正向预测与逆设计，结合 FP 腔（Fabry-Pérot）实现高饱和度色彩调控。

**云端演示**: [Streamlit Cloud](https://metasurface-web-a5mzzj8p98wur8njywdxat.streamlit.app/)

## 功能速览

| 功能 | 说明 |
|------|------|
| 🎨 实时预览 | 单柱/双柱纳米结构参数调节，实时显示颜色 |
| 🔍 网格搜索逆设计 | 扫描参数空间匹配目标颜色 |
| 🧠 ML 代理加速 | ONNX 神经网络模型，毫秒级色彩预测 |
| 📡 角谱远场传播 | NA/观察角对颜色的影响分析 |
| 🖼️ 图案生成 | 上传图片生成纳米柱阵列图案 |
| 🏗️ FP 腔设计 | DBR + 金属反射镜 Fabry-Pérot 腔高饱和色 |
| 🤖 AI 智能分析 | 接入 DeepSeek 大模型，智能解读颜色结果 |

## 技术架构

```
用户界面 (Streamlit)
    ├── 物理引擎 (Lorentzian 共振模型 / FP 腔)
    ├── ML 引擎 (ONNX Runtime 推理)
    ├── 逆设计 (网格搜索 + 梯度优化)
    └── LLM 模块 (DeepSeek API)
```

## 环境要求

- Python 3.10+
- Windows 10/11（Linux/macOS 需调整 `run.bat`）

## 快速开始

### 方式一：一键运行（推荐）
双击 `run.bat`，自动安装依赖并启动应用。

### 方式二：手动运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器访问 `http://localhost:8501`

### 启用 AI 分析（可选）
```bash
set DEEPSEEK_API_KEY=你的API密钥
streamlit run app.py
```

## 项目结构

```
metasurface-web/
├── app.py                  # Streamlit 主程序
├── ml_module.py            # MLP 模型 & ONNX 推理
├── torch_model.py          # PyTorch 物理+梯度逆设计
├── llm/                    # 大模型模块 (DeepSeek)
│   ├── deepseek_client.py  # API 客户端
│   └── prompts.py          # 提示词模板
├── models/                 # 训练好的模型权重 (.onnx / .pt)
├── training_data/          # 训练数据集 (.pkl)
├── train_*.py              # 模型训练脚本
├── run.bat                 # 一键运行脚本
├── requirements.txt        # Python 依赖
└── README.md               # 本文件
```

## 模型说明

| 模型 | 架构 | 输入 | 用途 |
|------|------|------|------|
| forward_mlp_v8_sub | ResMLP 256×4 | D, H, P, θ, pol, mat, sub | 单柱正向预测 |
| dual_mlp_v3_multi | DualResMLP | D1,H1,D2,H2,P,θ,pol,mat,sub | 双柱正向预测 |

## 成员分工

| 成员 | 职责 |
|------|------|
| 乔安琪 | 项目统筹、Streamlit 界面开发、LLM 模块集成 |
| 谢家珞 | 物理模型（Lorentzian/FP 腔）、角谱远场传播 |
| 侯琢 | ML 模型训练、ONNX 部署、逆设计算法 |
