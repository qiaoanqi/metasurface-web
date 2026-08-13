# OE 投稿检查清单（Optics Express Submission Checklist）

> 目标：商老师信息到位后，投稿 = 10 分钟填空作业。
> 图例：✅ 已就绪（零动作）｜⏳ 等导师/外部｜⚠️ 需人工在可访问网络下确认（国内直连 OPG 官网被 Radware 反爬拦截，curl/浏览器均不可达）

---

## A. 已就绪 ✅

| 项 | 位置 | 状态 |
|---|---|---|
| 论文正文锁定态 | paper_oe.tex（10 页、0 overfull、0 undefined；2026-08-13 relay #6 终锁：tex MD5=7368A098F5D8563AD31A747A51DAC442，PDF 双份 MD5=C6C69241EA687A1D3858386B0C646027） | ✅ |
| 图 6 张 | figures/*.pdf（fig1 protocol、fig2 判据、fig3 gamut、fig4 curse gap、fig5 nG、demo） | ✅ |
| Highlights 5 条 | highlights_optics_express.md | ✅ |
| 建议审稿人 4 主推 + 3 备选 | reviewer_candidates.md（从参考文献核实的名字/机构）；4 位已填入 cover letter | ✅ |
| Cover letter 正文 | cover_letter_optics_express.md（剩 2 处占位，见 B） | ✅ |
| 复现指南 | REPRODUCIBILITY.md（publication status: pending approval） | ✅ |

## B. 等导师（六项待办 → 填完即清零）⏳

1. 作者列表/署名/ORCID —— paper_oe.tex L16 `\author`（当前仅 Anqi Qiao + 通讯邮箱）
2. Author contributions —— paper_oe.tex L287 占位
3. Funding / 基金号 —— paper_oe.tex L280 占位
4. Data availability 决策 —— 现写 "upon reasonable request"，公开与否待拍板
5. 代码公开决策 —— 解除 REPRODUCIBILITY.md 的 "pending approval" 声明
6. Cover letter 共同作者署名 —— cover_letter L32 `[Co-author / advisor name...]` 占位

## C. 需人工在可访问网络下确认（校园网/VPN/浏览器手动过 CAPTCHA）⚠️

- [ ] 下载官方 LaTeX 模板 **optica-article.cls**（OE 官网 Author Instructions → 模板；被墙环境可从 LetPub VIP 模板页兜底）
- [ ] **摘要字数上限** —— Optica 通用约 350 words，OE 具体数值以官网 Author Guidelines 为准
- [ ] 正文页数建议 —— OE Research Article 无硬性上限，本项目 10 页应在安全区
- [ ] 参考文献格式细节 —— OPG 编号制（期刊缩写/卷/页码/年）；模板切换后逐条核对
- [ ] 图/表要求 —— 矢量 PDF 优先、分辨率/嵌入方式以官网为准
- [ ] APC 费用 —— OE 为 OA 期刊，金额以官网当期为准（学校财务可能需预审）
- [ ] 投稿系统确认 —— Optica 使用 Prism 平台（prism.optica.org），以官网为准
- [ ] 评审类型 —— 审稿人匿名（single-blind），以官网描述为准

## D. 提交前技术动作（模板切换时执行）

1. `article` → `optica-article.cls` 切换（等 B 项作者信息齐了一次做，避免排版回归返工）。模板已就位 `templates_optica/`（cls + jabbrv + styles/opticajournal.sty + 2×bst；来源：OPG 官网被 Radware 拦截，取自 GitHub 镜像 JoGruen/test 的 Optics_Express 目录；2026-08-08 隔离重编译两遍 exit=0，min_oe.pdf 61,792 B 双方各验一次）。切换三坑（已实测）：① `\title`/`\author` 必须在 `\begin{document}` 之后调用（OEtitle/OEauthor 是立即排版型）；② 必须 `\usepackage{microtype}`（OEtitle 内部用 `\textls`，cls 不自带）；③ `\journal{opticajournal}` 依赖 `styles/` 目录，缺失报 unsupported journal。迁移以 `templates_optica/min_oe.tex` 为骨架（正确 preamble），参照镜像 `main.tex` 结构
2. 填 L16 author / L280 Funding / L287 contributions
3. 双遍 `pdflatex -interaction=nonstopmode paper_oe.tex` → `Copy-Item paper_oe.pdf 论文.pdf -Force` → 核对 MD5 双份一致
4. 全文搜索占位符清零（`[to be added]`、`[Date]`、`[Co-author`）
5. Cover letter：填日期、更新 L18 竞业/基金声明、删 `[Emails to be confirmed...]`
6. **代码发布打包**（A1 遗留硬依赖）：`rcwa_batch.py` 对 `data/externals/aSi_PierceSpicer1972.csv` 是运行时硬依赖（缺失即 raise）——代码公开时必须随包发布 `data/externals/`；若 `app.py` / HF 部署链 import rcwa_batch，部署包同样要带此文件

## B2. 审稿人邮箱提取（唯一可提前清零的机械活，无需导师）⏳

网络受限说明：本环境 curl/浏览器均无法获取以下邮箱（OPG 被 Radware、Bing bot 无视查询词、DDG/Wayback 被墙），**不猜测邮箱**。提交前由人工从对应源论文通讯作者脚注或机构官方页复制，约 5 分钟：

| 审稿人 | 源论文（paper_oe.tex bib） | 提取处 |
|---|---|---|
| Alejandro W. Rodriguez | ref8 — Molesky et al., Nat. Photonics 12, 659 (2018) | 该文通讯作者脚注 / Princeton 官方页 |
| Ole Sigmund | ref26 — Christiansen & Sigmund, JOSA B 38, 496 (2021) | 该文通讯作者脚注 / DTU 官方页 |
| Haim Suchowski | ref24 — Malkiel et al., Light Sci. Appl. 7, 60 (2018) | 该文通讯作者脚注 / TAU 官方页 |
| Tie Jun Cui | ref28 — Cai et al., Appl. Phys. Lett. 119, 020501 (2021) | 该文通讯作者脚注 / 东南大学官方页 |

> OE 投稿系统也允许仅按姓名搜索审稿人，邮箱非必填——若时间紧可跳过此步直接提交。

## E. 数字口径提醒（防回归）

- Cover letter / Highlights 数字已按锁定态核对：TiO₂ 19%→62%、HfO₂ 82%、a-Si 86%@nG65（47%@nG101，双口径）、curse gap +2~4、Δn 截断 ~0.5、Si₃N₄ forward 13.75
- 论文标题以 paper_oe.tex L14 为准：**"...Optimizer's Curse and a Gamut Cutoff..."**（2026-08-06 已将 cover letter 旧版 "Gamut Limits" 对齐修正）
- 任何数字修改必须从 data/*.pkl 重提取核验，不凭记忆
