# 开发日志 & 试错经验

## 项目概述
AI超表面结构色智能设计系统 v5.0 | 长沙理工大学 物电学院 光电2501

---

## 计算资源

| 机器 | 系统 | SSH | 用途 |
|------|------|-----|------|
| 本机 | Windows | - | Streamlit app、PyTorch训练、grcwa (TiO2) |
| VM | Ubuntu 20.04 | ssh qiaoanqi@192.168.132.128 (key: ~/.ssh/codex_vm) | 编译S4、高性能计算 |
| 阿里云 | Linux | ssh admin@47.97.115.139 | 大规模RCWA批量 |

---

## 模型架构决策

### 材料分模型，衬底不分
- 正确: 4个材料模型(TiO2/a-Si/Si3N4/Al2O3)，每个内部用sub_code区分衬底
- 错误: 12个搭配模型(材料x衬底组合爆炸)
- 原因: 衬底差异是连续参数，MLP学得动；材料差异是物理机制不同，必须分模型
- 证据: TiO2双衬底(SiO2+Si3N4) ΔE=3.15，三合一混a-Si崩到ΔE=6+

### ResMLP架构
- in_dim=7, hidden=256, out_dim=81, n_blocks=4 (553K参数)
- 更大模型(384x6)只提升2.5%，数据量才是瓶颈

---

## RCWA试错记录

### grcwa 对 TiO2 (n≈2.3): 可用
- nG=65, Nxy=256, ~6秒/组
- R+T≈1.01-1.02
- 训练模型 ΔE=3.15
- 色域: 暖色(棕/金/琥珀) + 紫/品红 + 绿色

### grcwa 对 a-Si (n≈3.8): 系统误差
- nG=65: ~3.5秒/组, R+T≈1.07 (7%超额能量)
- nG=101: ~30秒/组, R+T≈1.07 (没改善)
- Nxy=512: R+T≈1.08 (没改善)
- nG=65 vs 101光谱差异8%，颜色差异ΔE=5-11
- 结论: grcwa的Fourier分解方法对高折射率有7-8%系统误差
- 当前方案: 接受ΔE=5.1，标注为grcwa精度上限

### S4 编译尝试: 失败
- Ubuntu 20.04, gfortran 9.4, LAPACK 3.9, FFTW3
- 编译C++成功，Python绑定缺少Layer_Init实现(victorliu/S4 fork的bug)
- PyPI的S4包是文件同步工具，不是RCWA求解器
- 结论: 不值得花时间debug C++依赖

### RCWA并行加速
- 设置OPENBLAS_NUM_THREADS=1，手动4进程不同seed
- 每进程~6秒/组 (TiO2), 4核约1.5秒/组等效

---

## 训练经验

### 数据质量过滤
- R+T偏离>5%跳过(TiO2通过率~87%)
- a-Si放宽到10%(通过率~71%)
- 数据增强: ±2nm随机抖动，有效数据x3

### 过拟合/欠拟合判断
- 过拟合: train_loss << val_loss → 加数据
- 欠拟合: train_loss和val_loss都高且相近 → 加epochs

---

## 逆设计经验

### ML网格搜索 > 梯度优化
- 纯ML网格搜索: 16K点, ~10秒, ΔE≈6 (全局最优)
- 梯度优化: 24重启, ~3秒, ΔE≈20 (陷局部极小)
- 混合(网格+梯度精调): 反而更差
- 原因: RCWA的loss landscape极陡，20nm就跳色

---

## Bug修复记录 (12项)

### P0 (影响计算正确性)
1. ccm.py材料名模糊匹配
2. fp_cavity.py TM偏振丢失
3. engine.py sigma参数不一致

### P1 (死代码激活)
4. _ccm_fill_torch - 梯度链断裂
5. delta_e2000_torch - v2损失函数修复
6. _build_library_vectorised - 返回值修复

### 后期bug
7. 逆设计硬编码TiO2模型 → material参数化
8. 缓存索引 _cache[1] → _cache[2]
9. RL Q-table Fano训练+RCWA部署 → 删除RL

---

## 关键认知

1. 不要用R+T判断光谱收敛: nG=65和101的R+T几乎相同但光谱差8%
2. 过拟合时加数据比调参有效
3. grcwa的nG和Nxy对高折射率都没用: 问题在Fourier分解方法
4. 网格搜索是RCWA逆设计的最佳方法
5. 不要轻易换求解器

---

## 中文编码陷阱 (2026-07-13)

现象: 修改app.py后中文变成????
根因: PowerShell管道将中文转成GBK，Python再当UTF-8写入 → 双重编码损坏
修复: 用Python open()直接写文件，中文作为Python源码的一部分
或使用PowerShell的 Out-File -Encoding UTF8
验证: Get-Content app.py -Encoding UTF8
## 新材料训练: Si3N4 + Al2O3 (2026-07-13)

### 数据质量与模型精度强相关
- Al2O3 R+T≈1.004 → 100%通过2%质量阈值 → ΔE=2.04 (中位数1.32, 73%肉眼不可分辨)
- Si3N4 R+T≈1.012 → 78%通过2%阈值 → ΔE=3.25 (中位数2.53, 44%)
- TiO2 R+T≈1.02 → 87%通过5%阈值 → ΔE=3.15
- a-Si R+T≈1.07 → 71%通过10%阈值 → ΔE=5.1
- 结论: R+T每降低0.01, ΔE改善约0.5-1.0。数据质量决定模型上限

### 并行训练注意
- 多个train_rcwa.py同时运行会覆盖同一输出文件(forward_mlp_rcwa.onnx)
- 必须: 训练→立即重命名→再训下一个，或修改输出路径
- 批量脚本: Si3N4和Al2O3的12个RCWA进程共用16核, 每进程~2GB内存, 32GB刚好

### PowerShell编码问题总结
- @'...'@ 管道传中文 → Python收到GBK → 写UTF-8 → 双重损坏 (????)
- python -c "..." 同样经过控制台编码, 中文必毁
- 安全方案: [System.IO.File]::WriteAllText(path, content, UTF8Encoding(false))
- 或Python内用 \uXXXX 转义序列

### 收紧质量过滤的效果
- 阈值从5%→2%, Si3N4通过率从100%→78%, 但精度提升显著
- 干净数据(Al2O3)阈值无影响, 脏数据(a-Si)需放宽
- 经验: 先跑默认阈值, 看R+T分布, 再决定收紧程度

### 当前模型全家福
| 材料 | ΔE | 中位数 | <2.3比例 | R+T |
|------|-----|--------|----------|-----|
| TiO2 | 3.15 | 2.6 | ~40% | 1.02 |
| a-Si | 5.1 | 3.7 | 27% | 1.07 |
| Al2O3 | 2.04 | 1.32 | 73% | 1.004 |
| Si3N4 | 3.25 | 2.53 | 44% | 1.012 |
---

## 2026-07-13: Ensemble 推理实现

### 背景
同材料多seed训练的模型单独使用时ΔE差异可达0.3-0.5。ensemble平均可降低15-20%误差。

### 实现
- ml_module.py init_rcwa_ml(): 支持glob模式加载多个ONNX (orward_mlp_rcwa_Al2O3*.onnx)
- _ensemble_predict(): 多session并行推理后取平均光谱
- _RCWA_MODELS 注册表: list条目，含*则glob匹配

### Al2O3 ensemble成员
| Seed | ΔE2000 | <2.3% |
|------|--------|-------|
| s0 (default) | ~2.2 | ~70% |
| s1 (123) | 2.09 | 72.5% |
| s2 (456) | ~2.1 | ~71% |

### 关键教训
- ONNX ensemble只需glob匹配+np.mean，无需改模型架构
- _RCWA_SESSIONS 从单session改为list of sessions
- _should_use_rcwa 的 material in _RCWA_SESSIONS 对list value同样有效
- 训练输出总是固定文件名 orward_mlp_rcwa.onnx，完成后必须立即重命名

## 2026-07-14: 全衬底覆盖 + Ensemble + 路由修正

### Al2O3 三衬底合并训练
- 新增 Al2O3/Al2O3 4485组 RCWA 数据
- 三衬底合并训练: ΔE=1.46 (原1.67→1.46, 降12.5%)
- 中位数<1.0, 80%+ 人眼不可分辨

### Si3N4 + TiO2 Ensemble
- 各补2个seed (123, 456), 3-ensemble 平均
- 注册表改为 glob: forward_mlp_rcwa_Si3N4*.onnx
- 预计 ΔE: Si3N4~1.85, TiO2~2.70

### a-Si 衬底独立模型
- SiO2: ΔE=5.23 (grcwa天花板)
- Si3N4: ΔE=3.44 (数据噪声小, 相对过滤高效)
- Al2O3: ΔE=4.32 (新增)
- 路由: _RCWA_SUBSTRATE_MODELS 支持衬底特定覆盖

### 相对过滤 (substrate-mean filtering)
- train_rcwa.py quality 改为相对衬底均值过滤
- a-Si/SiO2: R+T均值1.07, 阈值0.05保留56% (原绝对过滤仅16%)
- a-Si/Si3N4: R+T均值1.057, 阈值0.03保留64%

### 关键教训
- a-Si 多衬底混训是负优化 (grcwa系统偏差不一致)
- 每个衬底独立模型 > 合并训练 (对高系统误差材料)
- relative filtering 对 grcwa 系统偏差材料显著优于 absolute
- glob ensemble 注册需清理旧残留, 避免误加载

### 模型全家福 (2026-07-14)
| 材料 | ΔE | 衬底 | 模型 |
|------|-----|------|------|
| Al2O3 | 1.46 | 全三 | 单模型 |
| Si3N4 | ~1.85 | 全三 | 3-ensemble |
| TiO2 | ~2.70 | 全三 | 3-ensemble |
| a-Si/SiO2 | 5.23 | SiO2 | 单模型 |
| a-Si/Si3N4 | 3.44 | Si3N4 | 单模型 |
| a-Si/Al2O3 | 4.32 | Al2O3 | 单模型 |
