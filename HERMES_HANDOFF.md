# HERMES_HANDOFF.md — 论文2执行与自动接管交接（2026-08-15 实测版）

> 本文件按磁盘与进程实测填写，供 Codex"论文2执行与自动接管"任务完整接管。
> 这是 2026-08-15 01:31（北京时间）的交接快照；池生成阶段后来已完成。当前运行状态以 `.state/*.json`、`pipeline_policy.json` 和 `pipeline_supervisor.py` 为准，本文中的历史 PID/进度不覆盖磁盘实测。
> 所有路径相对项目根
> `C:\Users\A\Desktop\AI超表面结构色智能设计系统`。

---

## 1. 当前目标、阶段

**总目标**：论文 2（椭圆 × 双偏振超表面结构色，ML 逆设计）——诅咒框架在 4 维形状空间的可证伪检验。

| 阶段 | 状态 |
|---|---|
| 门 1 椭圆+偏振管道（W_nm/pol/采样器） | ✅ 完成（commit c09f154） |
| 门 2 L=W 复现门（30/30 逐位 0.0） | ✅ 完成（gate2_reproduction.py） |
| 门 4 nG 收敛探针（旧求解器） | ❌ 作废（错误求解器产物） |
| **求解器修复 + 背景口径** | ✅ 完成（commit 6671071/f84794a/803309a） |
| **门 4 nG 收敛门禁（空气背景）** | ✅ 完成（nG=131 达标，ng_gate_air_result.md） |
| **池生成（6000 条，nG=131，空气）** | ✅ 完成（严格池审计通过；见 `.state/audit_result.json`） |
| **联合收敛 v1 + v1.1** | 🔄 v1 运行中；v1.1 未启动（必须等 v1 400/400） |
| 训练（椭圆×双偏振 ML） | ⛔ 禁止（等池复核） |
| 闭环验证 + 论文 2 骨架 | ⏳ 后续 |

## 2. 当前池进程（实测）

- **启动命令**（完整）：
  ```
  python pool_generate_elliptical.py --samples 3000 --material TiO2 --pol both --nG 131 --n-jobs 16 --out data/rcwa_ellip_TiO2_3000_air.pkl
  ```
- **PID=42204，父 PID=4336**，启动 2026-08-15 01:16:34（Windows PowerShell 包装进程；实际 python 子进程在 multiprocessing 池内）
- **输出文件**：`data/rcwa_ellip_TiO2_3000_air.pkl`
- **当前记录数**：300/6000（实测于 01:30:57 落盘）
- **速度**：~20.8 条/分钟（300 条/14.4 min，16 核并行实测）
- **预计完成**：~2026-08-15 06:05（剩余 5700 条 × ~4.8 min/百条）
- **断点续跑**：`--resume` 读已有 pkl 的 (L,W,H,P,pol) done 集合，幂等跳过；增量落盘每 50 条（临时文件 + os.replace 原子写）
- **无需干预**：进程健康（CPU 活跃）；完成时自动通知

## 3. 已提交 commits（全部已推送远端 master）

| commit | 内容 | 验证 |
|---|---|---|
| `6671071` | 求解器修复：grcwa `RT_Solve(normalize=1)` + `GridLayer_geteps` 替代手动 kp0 求解链；周期网格 `endpoint=False` | 修复后 R+T=1.000000（能量守恒恢复）；gate2 复现 + 第三方裁判通过 |
| `f84794a` | 背景介质显式化：`background='air'`（默认，eps=1.0）/`'substrate'`（论文 1 冻结兼容）；`_rcwa_single_wl` 加 bg_eps | 双背景守恒 R+T=1.000000；air vs substrate dE=12.70（物理不同） |
| `803309a` | REPRODUCIBILITY.md：求解器修复记录 + 背景口径 + 论文 1 数据冻结声明 | — |

之前：`89d3878`（池生成器+gitignore 修正+口径记录）、`c09f154`（管道门 1-4）、`f7d2a58`（relay6-final tag）。

## 4. 求解器根因、修复、背景口径

- **根因**：`_rcwa_single_wl` 旧路径手动构造图案层 KP 矩阵——先用入射空气 kp0 求解（`SolveLayerEigensystem(..., kp0, eps2)`），再存图案层 kp（`MakeKPMatrix(..., epsinv_nG)`）——模式不一致 → **R+T>1**（P>~450 nm 出现，corr(P,R+T)≈0.63，max 1.12-1.91）。
- **修复**：删除手动链（Epsilon_fft/MakeKPMatrix/Solve*/GetZPoyntingFlux），改用 `blk.GridLayer_geteps(eps_grid.ravel())` + `blk.RT_Solve(normalize=1)`（grcwa 官方路径）。
- **背景口径（审计 2026-08-14 裁决）**：论文 2 = **柱间空气**（`background='air'` 默认，eps=1.0，纳米柱立在衬底上柱间为空气）；论文 1 历史数据 = 柱间衬底（`'substrate'`，eps=n_sub²）——冻结不改。

## 5. nG=131 门禁依据（完整）

- **探针文件**：`data/ng_probe_elliptical_TiO2_air.pkl`（16 个椭圆结构 × nG {65,101,131,201} × pol {p,s}，空气背景 + 修复求解器）
- **脚本**：`probe_ng_elliptical.py --material TiO2 --out data/ng_probe_elliptical_TiO2_air.pkl`
- **判据（预注册，未放宽）**：以最高阶 nG=201 为参考，每结构联合 `max(ΔE_p, ΔE_s)`；选 **mean ΔE < 0.5 JND（=1.15）的最小 nG 档**。
- **原始数字**（以 201 为参考）：
  | nG | mean ΔE | median | <1JND | <2.3JND |
  |---|---|---|---|---|
  | 65 | 1.83 | 1.70 | 12% | 69% |
  | 101 | 1.08 | 1.00 | 50% | 94% |
  | **131** | **0.46** | 0.40 | **94%** | **100%** |
  | 201 | 0（参考） | — | — | — |
- **结论**：nG=131（11×11 谐波）达标 → **池定 nG=131**。旧结论（衬底背景+旧求解器：131 mean 1.45 不达标需 13×13）作废——错误求解器产物。
- 耗时实测：nG=131 = 92.9 s/结构/偏振。

## 6. 论文 1 暴露、shadow、冻结、终锁

- **暴露成立**：`data/rcwa_5k.pkl`（07-07）+ `rcwa_5k_b2.pkl`（07-09）的 R+T vs P 与当前池同型破守恒（P[450,500) mean 1.048、max 1.12）——旧求解路径从源头（07-07）就有 bug，论文 1 全部数据受影响。
- **shadow 影响评估**（`data/paper1_impact_shadow.pkl`，脚本 `paper1_impact_shadow.py`，8 结构 P 200-499）：修复后（同衬底背景）光谱 max|ΔR| 0.23-0.96、dE 中位 12.76；air 口径 dE 中位 16.36；修复后全守恒。
- **冻结边界**：论文 1 原始池/论文/PDF/统计结果**零改动**（rcwa_5k.pkl mtime 07-07 09:20、rcwa_5k_b2.pkl 07-09 06:55 实测未变）；不再称为物理有效数据；统计相对性是否保留待评估。
- **论文 1 终锁 MD5（实测）**：tex = `7368A098F5D8563AD31A747A51DAC442`；paper_oe.pdf = 论文.pdf = `C6C69241EA687A1D3858386B0C646027`。
- **论文 1 完整重跑**：不自动执行，独立重大决策。

## 7. 数据/脚本/审计文件路径

**关键数据**：
- `data/rcwa_ellip_TiO2_3000_air.pkl` — 新空气池（生成中，6000 条目标；记录字段 L/W/H/P/r/pol/nG_actual/R(81)/T(81)/xyz/rgb/R_plus_T_mean/quality_pass/isolated/time_s/retry_nG）
- `data/rcwa_ellip_TiO2_3000.pkl` — ⛔ 隔离旧池（衬底+旧求解器，禁止训练侧使用）
- `data/ng_probe_elliptical_TiO2_air.pkl` / `data/ng_probe_elliptical_aSi.pkl` / `data/ng_probe_elliptical_TiO2.pkl` — nG 探针
- `data/ellipse_gate1_invariants.pkl` / `data/ellipse_gate2_result.pkl` — 门 1/门 2 产物
- `data/paper1_impact_shadow.pkl` — shadow 影响评估
- `data/rcwa_5k.pkl` / `rcwa_5k_b2.pkl` — 论文 1 旧池（冻结，零改动）

**脚本**（均入库）：`pool_generate_elliptical.py`、`probe_ng_elliptical.py`、`gate1_invariants.py`、`gate2_reproduction.py`、`paper1_impact_shadow.py`（新，未入库）、`rcwa_batch.py`（核心求解器，已修复）。

**审计/结果 md**：`ng_gate_air_result.md`（新，未入库）、`ellipse_pool_result.md`（新，未入库）、`ellipse_gates_1_2_result.md`、`ng_probe_elliptical_result.md`、`second_paper_elliptical_plan.md`（含口径记录）、`REPRODUCIBILITY.md`（含修复/冻结声明）。

## 8. `.state/hermes_status.json` 状态协议

历史快照：`stage=pool_generation, status=running, pid=42204`。当前不要依据该行判断运行状态；请读取 `.state/hermes_status.json`、`.state/executor_ack.json` 和 `.state/controller_state.json`。

**池完成后必须原子更新为**：
```json
{
  "schema_version": 2,
  "stage": "pool_generation",
  "status": "completed",
  "outputs": [{"path": "data/rcwa_ellip_TiO2_3000_air.pkl", "material": "TiO2"}],
  "paper_hashes": [
    {"path": "paper_oe.tex", "md5": "7368A098F5D8563AD31A747A51DAC442"},
    {"path": "paper_oe.pdf", "md5": "C6C69241EA687A1D3858386B0C646027"},
    {"path": "论文.pdf", "md5": "C6C69241EA687A1D3858386B0C646027"}
  ],
  "next_on_pass": "pool_validation",
  "checks": {"pool_sha256": "<磁盘实测 SHA256>", "records": 6000},
  "constraints": {"no_training": true, "paper1_files_zero_change": true}
}
```
完成回执必须使用对象数组、包含 `checks.pool_sha256`，并由监督器重新计算所有文件哈希；原子写使用永久脚本的有限重试实现。

## 9. 禁令（全部生效）

1. **禁止训练**（池复核通过前不写训练脚本）
2. **禁止覆盖或 `--resume` 旧隔离池**（`rcwa_ellip_TiO2_3000.pkl`）
3. **禁止修改论文 1**（tex/pdf/统计/旧池零字节）
4. **禁止用临时验证脚本制造 changed-path 循环**（已确认系统扫描机制缺陷；用实测定点命令替代临时脚本）

## 10. 已知风险、未验证事项、下一步

**风险**：
- 新池 R+T 分布未知（旧池偏 1.03；修复后单结构 1.000000，池级待实测）
- quality_pass 率未知（绝对判据 |R+T−1|≤0.05）
- 16 核 8P+8E 中 E 核拖后腿（实际吞吐 ~20.8 条/min 已实测）
- Windows 睡眠/断电中断（断点续跑已覆盖）

**未验证**：新池统计五样（总耗时/ok-fail-isolated/quality_pass/R+T 分布/隔离数）；训练侧配对逻辑（pass 为单偏振级）。

**建议下一步（按序）**：
1. 池严格核验与不可变清单（已通过）
2. D65 色度学门禁（已通过）
3. 联合收敛 v1 完成后运行 v1.1（400+64 任务，1 nm 完整积分）
4. 同参数跨求解器光谱门禁
5. 圆柱对照、轴/偏振规范化和 geometry split 冻结
6. 只有 controller 与 audit 两处 `training_allowed=true` 才能运行训练 pilot
7. 闭环矩阵与论文 2 最终结果审计；失败则缩窄结论，不改阈值

## 11. 工作树未提交文件（归属，禁止误删）

| 文件 | 归属 | 说明 |
|---|---|---|
| `M rcwa_batch_fast.py` | 历史遗留 | 独立求解器（rcwa 包），未参与修复，暂不动 |
| `?? .state/` | **Codex + Hermes** | 监督器（paper_index.json/pipeline_supervisor.py 等）+ hermes_status.json |
| `?? paper_index.json/md`、`pipeline_status.example.json`、`pipeline_supervisor.py` | **Codex** | 监督器文件，勿动 |
| `?? data/ng_probe_elliptical_TiO2_air.pkl` | Hermes | nG 门禁探针（data/* ignore，可留磁盘） |
| `?? ellipse_pool_result.md`、`ng_gate_air_result.md`、`paper1_impact_shadow.py` | Hermes | 新结果/脚本，池完成后随 commit 入库 |
| `?? _archive/`、`scripts/` | 历史遗留 | 归档目录，勿删 |

## 12. 交接语句

**从本文件写完起，后续执行权交给 Codex"论文2执行与自动接管"任务。Hermes 不再启动新阶段（池生成完成后仅做状态原子更新与五样报告），除非用户重新授权。**
