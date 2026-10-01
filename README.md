# AI超表面结构色智能设计系统

面向竞赛展示的超表面结构色设计与可视化系统。当前主线是单柱参数探索、光谱/色度表达、已审核 TiO₂/SiO₂/air 参考复核和结果导出；其他结构与材料路线按交互页的来源标注运行。

## 当前入口

- 在线展示页：<http://47.111.14.70/>
- 在线交互页：<http://47.111.14.70/app/>
- GitHub 代码仓库：<https://github.com/qiaoanqi/metasurface-web>
- 系统文档与资源索引：[competition/13_公开资源索引.md](competition/13_公开资源索引.md)
- 离线包：<https://github.com/qiaoanqi/metasurface-web/releases/download/competition-web-v1.0.0/ai_metasurface_web_bundle_v1.zip>

二维码当前直接指向在线交互页 `/app/`。公网 IP 变更后，需要同步更新二维码、网站链接和本文件中的地址。

报名系统中的队伍身份信息仍单独填写，公开仓库只保留竞赛运行时、展示页和脱敏说明。

---
## 功能概览

| 模块 | 功能 |
|------|------|
| 🎨 实时预览 | 纳米柱参数 → 反射光谱 → sRGB颜色，按当前路线更新 |
| 🔍 逆设计 | 目标颜色 → 当前搜索返回的候选参数（网格搜索 / RL / 梯度路线） |
| 🖼 图案生成 | 上传图片 → 逐像素匹配纳米柱 → 超表面阵列可视化 |
| 🗺 色域映射 | CIE 1931 色度图上叠加 sRGB 色域、材料色域边界对比 |
| 📊 光谱分析 | 反射光谱曲线、入射角扫描、偏振对比 |
| 🧭 路由提示 | 按结构、材料、衬底和入射条件提示当前可用路线 |
| 🔬 远场传播 | 角谱理论 + NA 锥积分，模拟人眼/显微镜观察效果 |
| 📦 数据导出 | 光谱 CSV、色板 PNG、逆设计结果 JSON 一键下载 |

**路线范围：** 单柱是当前比赛主线；双柱、FP 腔、多材料和远场功能按界面显示的模型状态运行。  
**结果边界：** 颜色与光谱属于当前路线内的候选表达，不直接等同于实验测量或全域物理精度。

---

## 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 启动竞赛交互台

```bash
python -m pip install -r requirements-web.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8512
```

浏览器打开 `http://127.0.0.1:8512/`。Windows、Linux 和 macOS 的完整说明见 [competition/11_本地离线演示说明.md](competition/11_本地离线演示说明.md)。

---

## 目录导航

```
├── app.py                              # Streamlit 交互入口
├── engine.py / ml_module.py            # 前向路线与代理模型
├── color_utils.py                      # CIE 1931 / Lab / ΔE2000
├── competition/                        # 竞赛文档、参考库和展示页
├── competition/website展示页_校赛副本/ # 网站式展示页
├── deployment/                         # ECS、Nginx 和 systemd 部署文件
├── scripts/                            # 发布包、manifest 和健康检查脚本
├── requirements-web.txt                # 竞赛运行时依赖
└── dist/                               # 本地生成的发布包（ZIP 不进 Git）
```

---

## 运行边界

竞赛包默认关闭外部大模型功能，不需要 API 密钥。页面中的代理预测、解析路线和参考复核会分别标注来源；参考库只对已收录整数几何做精确命中，不做最近邻或插值。

---

## 云端部署

### 当前阿里云入口

- 展示页：<http://47.111.14.70/>
- 交互页：<http://47.111.14.70/app/>
- 健康检查：<http://47.111.14.70/health>

部署、更新和回滚命令见 [deployment/DEPLOY_TO_ECS.md](deployment/DEPLOY_TO_ECS.md)。

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 物理引擎 | Lorentz/Fano共振 · CCM耦合补偿 · 米氏散射 · FP腔TMM · 角谱远场传播 |
| ML 加速 | ResMLP · ONNX Runtime · 运行时间随设备与路线变化 |
| 逆设计 | 网格搜索 · Q-Learning RL · PyTorch 梯度优化 |
| 色度学 | CIE 1931 · CIEDE2000 · sRGB · ConvexHull 色域 |
| 前端 | Streamlit · Matplotlib · 纯CSS纳米柱渲染 |
| 部署 | ONNX Runtime · Alibaba Cloud · Nginx + Streamlit |

---

## 发布资源

- 资源索引：[competition/13_公开资源索引.md](competition/13_公开资源索引.md)
- 本地交付清单：[competition/12_校赛本地交付清单.md](competition/12_校赛本地交付清单.md)
- ECS 部署说明：[deployment/README.md](deployment/README.md)
- 离线包：<https://github.com/qiaoanqi/metasurface-web/releases/tag/competition-web-v1.0.0>

公网可访问只表示竞赛展示服务正常，不代表科研控制面或实验验证已经完成。
