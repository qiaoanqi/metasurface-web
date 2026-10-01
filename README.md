# AI超表面结构色智能设计系统

面向超表面结构色验证场景的 AI 设计与可视化工作流。当前竞赛主线是本地可复核的单柱参数探索、光谱/色度表达、已审核参考复核和结果导出；双柱、FP 腔及其他材料属于辅助演示路线，具体可用范围以交互页的来源标注为准。

公网地址、代码仓库和队伍身份信息在提交前单独配置，不在脱敏项目说明中固定历史地址或个人信息。

---
## 功能概览

| 模块 | 功能 |
|------|------|
| 🎨 实时预览 | 纳米柱参数 → 反射光谱 → sRGB颜色，按当前路线更新 |
| 🔍 逆设计 | 目标颜色 → 当前搜索返回的候选参数（网格搜索 / RL / 梯度路线） |
| 🖼 图案生成 | 上传图片 → 逐像素匹配纳米柱 → 超表面阵列可视化 |
| 🗺 色域映射 | CIE 1931 色度图上叠加 sRGB 色域、材料色域边界对比 |
| 📊 光谱分析 | 反射光谱曲线、入射角扫描、偏振对比 |
| 🧠 AI 分析 | DeepSeek 大模型解读颜色物理机理 + 参数优化建议 |
| 🔬 远场传播 | 角谱理论 + NA 锥积分，模拟人眼/显微镜观察效果 |
| 📦 数据导出 | 光谱 CSV、色板 PNG、逆设计结果 JSON 一键下载 |

**路线范围：** 单柱是当前比赛主线；双柱、FP 腔、多材料和远场功能按界面显示的模型状态运行。  
**结果边界：** 颜色与光谱属于当前路线内的候选表达，不直接等同于实验测量或全域物理精度。

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

### 阿里云（提交前配置）

公网地址、端口和 HTTPS 状态以最终部署复核记录为准。推荐校赛展示实例为 4 vCPU / 16 GB RAM / 100 GB SSD；本地离线包作为公网故障时的备份。

### HuggingFace Spaces（备选）

`qiaoanqi/metasurface-color-designer`

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 物理引擎 | Lorentz/Fano共振 · CCM耦合补偿 · 米氏散射 · FP腔TMM · 角谱远场传播 |
| ML 加速 | ResMLP · ONNX Runtime · 运行时间随设备与路线变化 |
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
