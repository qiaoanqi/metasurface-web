# AI超表面结构色智能设计系统 v5.0

> **长沙理工大学 · 物理与电子科学学院 · 光电2501班**
> 乔安琪（组长）、谢家珞、侯琢 | 指导教师：甘文

基于物理引擎（Lorentz/Fano共振 + 耦合补偿模型CCM + FP腔TMM）与深度学习（ResMLP代理模型）的超表面纳米柱结构色正向预测与逆设计一体化平台。CCM通过数据驱动对麦克斯韦近似解残差修正，将光谱平均误差降低至2.1%以下。


**🟢 在线体验：http://47.97.115.139（展示页）
> Streamlit 演示：http://47.97.115.139:8080
> GitHub 仓库：https://github.com/qiaoanqi/metasurface-web**

---
## 功能概览

| 模块 | 功能 |
|------|------|
| 🎨 实时预览 | 纳米柱参数 → 反射光谱 → sRGB颜色，毫秒级实时更新 |
| 🔍 逆设计 | 目标颜色 → 最优纳米柱参数（网格搜索 / RL / 梯度优化 三种算法） |
| 🖼 图案生成 | 上传图片 → 逐像素匹配纳米柱 → 超表面阵列可视化 |
| 🗺 色域映射 | CIE 1931 色度图上叠加 sRGB 色域、材料色域边界对比 |
| 📊 光谱分析 | 反射光谱曲线、入射角扫描、偏振对比 |
| 🧠 AI 分析 | DeepSeek 大模型解读颜色物理机理 + 参数优化建议 |
| 🔬 远场传播 | 角谱理论 + NA 锥积分，模拟人眼/显微镜观察效果 |
| 📦 数据导出 | 光谱 CSV、色板 PNG、逆设计结果 JSON 一键下载 |

**物理引擎覆盖：** 单柱 / 双柱 / FP 腔三种结构 × TiO₂ / a-Si / Si₃N₄ / Al₂O₃ 四种材料 × SiO₂ / Si₃N₄ / Al₂O₃ 三种衬底  
**CCM 耦合补偿：** 数据驱动修正 f_eff = f₀ + Δf(L,W)，FDTD 标定 6 组材料组合系数，光谱误差 < 2.1%

---

## 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 一键运行

**Windows：** 双击 `run.bat`（首次自动安装依赖）

**Linux/macOS：**
```bash
chmod +x run.sh
./run.sh
```

### 手动安装

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

浏览器打开 `http://localhost:8501`

---

## 目录结构

```
├── app.py              # Streamlit 主程序
├── engine.py           # 物理引擎（Lorentz/Fano共振 + 逆设计搜索）
├── torch_model.py      # PyTorch 批量物理模型 + 梯度逆设计
├── ml_module.py        # ML 代理模型（ResMLP ONNX推理 + numpy梯度优化）
├── fp_cavity.py        # FP 腔传输矩阵法（金属镜 / DBR介质镜）
├── rl_design.py        # Q-Learning 强化学习逆设计
├── ccm.py              # 耦合补偿模型 (CCM): f_eff = f0 + Δf(L,W)
├── color_utils.py      # CIE 1931 色度学工具（XYZ/Lab/ΔE2000）
├── llm/                # 大模型模块（DeepSeek API）
├── models/             # ONNX 模型权重 + PyTorch checkpoint
│   ├── forward_mlp_v8_sub.onnx    # 单柱 ML 模型（含衬底编码）
│   ├── dual_mlp_v3_multi.onnx     # 双柱 ML 模型
│   └── rl_qtable.pkl              # RL Q表
├── data/               # 数据集与预处理
├── requirements.txt    # Python 依赖
├── run.bat / run.sh    # 一键运行脚本
├── .env.example        # 环境变量模板
└── README.md
```

---

## 环境配置

复制 `.env.example` 为 `.env`，填入 API 密钥：

```ini
DEEPSEEK_API_KEY=你的DeepSeek密钥
HF_TOKEN=你的HuggingFace令牌（可选）
```

- `DEEPSEEK_API_KEY`：用于 AI 智能分析功能，不填则 AI 分析不可用
- `HF_TOKEN`：用于从 HuggingFace Hub 自动下载模型，本地已有 models/ 则无需

---

## 可选依赖

| 依赖 | 用途 | 安装命令 |
|------|------|---------|
| PyTorch | 梯度逆设计、RL训练、批量色卡、灵敏度分析 | `pip install torch` |

不装 PyTorch 时，系统会自动降级为纯 numpy/ONNX 路径，核心功能不受影响。

---

## 云端部署

### 阿里云（当前线上地址）

**http://47.97.115.139**

轻量应用服务器 · 2 vCPU / 4GB · Nginx 展示页 :80 + Streamlit 演示 :8080

### HuggingFace Spaces（备选）

`qiaoanqi/metasurface-color-designer`

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 物理引擎 | Lorentz/Fano共振 · CCM耦合补偿 · 米氏散射 · FP腔TMM · 角谱远场传播 |
| ML 加速 | ResMLP (256×4) · ONNX Runtime · 推理 5.13 ms |
| 逆设计 | 网格搜索 · Q-Learning RL · PyTorch 梯度优化 |
| 色度学 | CIE 1931 · CIEDE2000 · sRGB · ConvexHull 色域 |
| 前端 | Streamlit · Matplotlib · 纯CSS纳米柱渲染 |
| LLM | DeepSeek Chat API · 定制 Prompt 工程 |
| 部署 | ONNX Runtime · Alibaba Cloud · Nginx + Streamlit · HuggingFace Spaces |

---

## 项目报告

详见 `AI超表面结构色设计_项目报告.docx`（含完整技术文档、测试数据、参考文献）。

## 致谢

- 指导教师甘文老师
- 长沙理工大学物理与电子科学学院
- CIE 015:2018 色度学标准
- DeepSeek 大模型 API
