import os
os.chdir(r"C:\Users\A\Desktop\AI超表面结构色智能设计系统")

with open("_archive/old_drafts/LESSONS.md", "r", encoding="utf-8") as f:
    old = f.read()

new_section = '''
---

## 论文写作阶段 (2026-07-15 ~ 07-25)

### 核心发现：闭环验证揭示优化器诅咒
- 领域惯性：ML超表面论文只报test-set ΔE，暗示"可用于逆设计"，无人做RCWA复验
- 我们做了：227个目标逐个RCWA重算，发现ML自称0.66→实测5.83（9倍落差）
- 这是论文存在的理由——不是方法新，是验证标准新

### 数据报告的血泪教训
**mean vs median 混用差点毁了Table 4。** a-Si N=29 curse gap：mean=+4.65, median=+1.87。Table标的是+4.65但列在median栏。审稿人自己跑一遍median就发现猫腻。现在：全用median，小样本加dagger脚注。
**目标类型不能望文生义。** 旧TiO₂ N=30 pkl里是rt_00~rt_29——roundtrip目标，不是gamut-probing。63%是"色域内成功率"，不是"全sRGB成功率"。发现时论文已写了两周。改完后多了一个完整的2×2矩阵（材料×目标类型），叙事反而更干净。
**RCWA计时数字自相矛盾。** 论文写4.16s，同代码实测1.63s（2.5倍差）。450候选也是假的（实际1392）。22×加速是两个假数相除凑出来的。重跑hybrid闭环拿实测数（ML ~4s + RCWA ~33s = ~37s，加速~60×），所有数字互洽。

### 论文结构重构
**两篇论文订在一起。** 优化器诅咒+色域极限是一条线，Δn共振截止是另一条。最终保留但压缩Δn，主线定在"色域密度天花板"。
**hybrid从"贡献"降级为"验证策略"。** ML筛→RCWA验是代理辅助优化的标准操作，数学上trivial。改为diagnostic tool（诊断候选池质量），把重心移到"色域才是瓶颈"。
**nG收敛从Results移到Methods。** 那是质控不是发现。
**gamut collapse→candidate pool depletion。** a-Si色域范围与TiO₂相当（凸包24% vs 20%），但候选点聚成一团（sRGB std=0.1）。不是"做不到"，是"分布极不均匀"。这个区分在文献里没人讲过。

### 图的质量统一
- 独立脚本（fig2_12pts.py, fig4_4group.py）用serif+dpi=150，_make_figures.py用sans-serif+dpi=300
- 统一到_make_figures.py一套脚本出全部图，避免字体/dpi不一致
- Codex不碰图——DS API无视觉能力，看图烧钱且无意义。图的事全交Qwen

### Qwen-Codex 分工模式
**Qwen（有眼睛）：** Figure目检、排版美观度判断、叙事结构设计、审稿人视角模拟、英文化、措辞打磨
**Codex（有手）：** 跑RCWA数据、编译tex、grep验证数字一致性、pkl验算、K敏感性分析、CrossRef文献补全、文件操作
**铁律：Codex永远不用view_image。** 烧了10块钱看图——像素数据对DS API毫无意义。

### 关键数字决策
- N=30→N=100：TiO₂ roundtrip 63%→62%，Wilson CI从[44%,79%]收窄到[52%,71%]
- K=20是最优膝点：K=5→10→20→50对应47%→57%→62%→64%，K=20后边际收益从+0.5%/call降至+0.07%/call
- a-Si mean predictor 19.6 vs ML 2.38（8×提升）→精度是真本事，逆设计仍0%→瓶颈不是模型是色域
- Si₃N₄ mean predictor 3.4 vs ML 1.57（2×提升）→精度是trivial的，根本没结构可学

### tex损坏恢复
- 替换操作匹配到Abstract里的"This paper makes three contributions"而非Introduction里的，删掉了整个preamble
- 从备份体逐段提取preamble/title/abstract/Introduction，手工重组
- 教训：全局搜索替换必须先确认匹配次数和位置

### 审稿人防御
- 加极端值理论解析下界：E[gap]≈σ√(2ln N)，代入σ≈3, N=1392→√(2ln1392)≈3.8，预测3–5ΔE，实测+3.59吻合
- 加可证伪预测：Δn>0.6+k≈0+K=20→成功率predicted to exceed 50%
- 加实操指南四条（Δn预筛、闭环验证、K≥20、<30%换材料不换模型）
- "never/none"→"to our knowledge"（避免一篇反例文献就打脸）
- "universal"→"empirical, within the investigated parameter space"
- ref19-21（2025-26）证明领域仍活跃但无人做过闭环验证

### 当前状态
- 11页，21条引用，5图4表，0 errors
- 待作者信息→cover letter→OE模板切换→投稿
'''

with open("LESSONS.md", "w", encoding="utf-8") as f:
    f.write(old + new_section)

print("LESSONS.md updated. Old preserved in _archive/old_drafts/")
