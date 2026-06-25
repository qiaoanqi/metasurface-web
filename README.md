# AI 超表面结构色智能设计系统 v5.0

> 长沙理工大学 物电学院 光电2501
> 乔安琪、谢家珞、侯琢
> 《人工智能基础B》期末大作业

## 项目简介

基于深度学习和物理模型的超表面纳米柱结构色智能设计平台。支持 TiO2、a-Si、Si3N4、Al2O3 四种材料与 SiO2、Si3N4、Al2O3 三种衬底的单柱/双柱纳米结构正向预测与逆设计，集成 FP 腔实现高饱和度色彩调控。

**云端演示**: https://huggingface.co/spaces/qiaoanqi/metasurface-color-designer

## 目录结构

```
├── app.py                  # Streamlit 主程序
├── engine.py               # 物理引擎 + 逆设计引擎
├── torch_model.py          # PyTorch 物理模型 + 梯度逆设计
├── ml_module.py            # MLP 模型 & ONNX 推理
├── fp_cavity.py            # FP 腔光谱计算
├── color_utils.py          # CIE 1931 色度学 + CIEDE2000
├── rl_design.py            # Q-learning 强化学习逆设计
├── llm/                    # 大模型模块
│   ├── deepseek_client.py  # HF Inference API 客户端
│   └── __init__.py
├── models/                 # 训练好的 ONNX 权重
│   ├── forward_mlp_v8_sub.onnx
│   ├── forward_mlp_v7_multi.onnx
│   ├── dual_mlp_v3_multi.onnx
│   └── rl_qtable.pkl
├── fdtd_data/              # FDTD 验证数据与对比图
├── requirements.txt        # Python 依赖（精确版本号）
├── run.bat                 # Windows 一键运行
├── run.sh                  # Linux/macOS 一键运行
└── README.md               # 本文件
```

## 环境要求

- Python 3.10+
- Windows 10/11 或 Linux/macOS

## 快速开始

### Windows
双击 `run.bat`，自动安装依赖并启动应用。

### Linux / macOS
```bash
chmod +x run.sh
./run.sh
```

### 手动运行
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

浏览器访问 http://localhost:8501

## API Key 配置（可选）

AI 智能分析功能需要 Hugging Face API Token：

**Windows**:
```cmd
set HF_TOKEN=你的HF令牌
streamlit run app.py
```

**Linux/macOS**:
```bash
export HF_TOKEN=你的HF令牌
streamlit run app.py
```

也可在项目根目录创建 `.env` 文件：
```
HF_TOKEN=你的HF令牌
```

## 功能速览

| 功能 | 说明 |
|------|------|
| 预览 | 单柱/双柱/FP腔实时颜色预览 |
| 逆设计 | 5种方法：网格搜索、RL、单柱梯度、双柱梯度、三方案对比 |
| 图案生成 | 上传图片生成纳米柱阵列图案 |
| 映射 | D-H颜色映射 + CIE 1931色度图 + 4体系色域对比 |
| 光谱 | 反射光谱 + 入射角扫描 + FDTD验证 |
| AI分析 | 大模型智能色彩解读与参数优化建议 |