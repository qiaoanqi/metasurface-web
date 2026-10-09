import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const out = path.join(root, 'competition', '答辩材料_20261009');
const version = process.argv[2] || 'v21';
const handbook = version === 'v18' ? '答辩准备手册_v1.md' : '答辩准备手册_v2.md';
const prep = JSON.parse(await fs.readFile(path.join(out, '答辩准备内容.json'), 'utf8'));
const slides = JSON.parse(await fs.readFile(path.join(root, '.defense-build-20261009', version === 'v18' ? 'slide_content.json' : `slide_content_${version}.json`), 'utf8'));
if (slides.length !== 15 || slides.reduce((sum, s) => sum + s.seconds, 0) !== 560) {
  throw new Error('Unexpected slide count or timing');
}
const paragraphs = items => items.map(v => v.replaceAll('—', '-')).join('\n\n');
const lines = [
  '# AI超表面结构色智能设计系统答辩准备手册',
  '',
  '主讲人：乔安琪。团队：404 Not Found队。成员：乔安琪、陈雍杰、郭千弘。',
  '',
  `本手册供团队内部排练与提交准备使用，不作为匿名作品附件直接上传。逐页讲稿对应答辩修订版 ${version}；PPT 已包含同版演讲者备注。`,
  '',
  '## 一、比赛安排与成员分工',
  '',
  paragraphs(prep.overview),
  '',
  '## 二、完整逐页讲稿',
  '',
  '建议总用时 9 分 20 秒。这是建议分配，尚未完成真人计时。首次排练读全稿，正式答辩看标题与关键词讲，不必把风险说明逐字重复。',
  '',
  '| 页 | 主题 | 建议秒数 | 累计时间 |',
  '| --- | --- | ---: | --- |',
];
let elapsed = 0;
for (const s of slides) {
  elapsed += s.seconds;
  lines.push(`| ${s.slide} | ${s.title.replaceAll('\n', ' ')} | ${s.seconds} | ${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, '0')} |`);
}
for (const s of slides) {
  lines.push('', `### 第 ${s.slide} 页 ${s.title.replaceAll('\n', ' ')}`, '', `建议用时：${s.seconds} 秒。`, '', s.script);
}
lines.push('', '## 三、约七分钟压缩讲稿', '', '该版本建议总用时 6 分 55 秒。时间紧时删减背景和软件概述，保留算法、真实候选、验证范围及创新定位。');
for (const s of prep.short_version) {
  lines.push('', `### 第 ${s.slides.replaceAll('–', '-')} 页 ${s.seconds} 秒`, '', s.script);
}
lines.push('', '## 四、问答卡', '', '先直接回答，再给证据和必要边界。证据路径供内部定位，不必在口头回答中念出。');
prep.qa.forEach((item, i) => {
  lines.push('', `### ${i + 1}. ${item.q}`, '', item.a, '', `内部证据：${item.evidence}`);
});
lines.push('', '## 五、排练计划', '', paragraphs(prep.rehearsal));
lines.push('', '## 六、备用演示与故障处理', '', paragraphs(prep.demo));
lines.push('', '## 七、视频要求与录制脚本', '', paragraphs(prep.video_rules));
for (const item of prep.video) {
  lines.push('', `### ${item.time.replaceAll('–', '-')} ${item.action}`, '', item.speech);
}
lines.push('', '## 八、提交检查清单', '', '以下清单不代表已经上传。正式答辩材料与本内部手册分别管理。');
for (const item of prep.checklist) {
  lines.push('', `- [${item.startsWith('已准备：') ? 'x' : ' '}] ${item}`);
}
lines.push('', '## 九、目前仍需补齐的内容', '', paragraphs(prep.missing));
lines.push('', '## 十、证据口径速查', '',
  '| 数字或结果 | 可支持的结论 | 不能推导的结论 |',
  '| --- | --- | --- |',
  '| 21,088 条，21,083 组独立几何 | 限定条件的参考数据审核 | 全材料数据库、新模型训练完成 |',
  '| 最大点级 R+T 守恒误差约 1.79e-7 | 参考记录内部一致性 | RCWA 全面收敛、实验精度 |',
  '| 最大 sRGB 分量复算差约 1.47e-8 | 存储颜色与独立复算一致 | 目标色差、代理准确率 |',
  '| 第一候选色差约 21.67 | 当前存档的真实候选及明显匹配差距 | 高精度设计成功 |',
  '| 27 项运行检查，19 项应用流程 | 2026-10-06 存档的软件执行检查 | 本轮重新测试、物理准确率 |',
  '| 双柱解析基线可运行 | 独立解析功能保留 | 双柱 ONNX 已发布 |',
  '', '## 十一、内部证据来源', '', paragraphs(prep.references), '');

await fs.writeFile(path.join(out, handbook), lines.join('\n'), 'utf8');
console.log(JSON.stringify({ output: path.join(out, handbook), slides: slides.length, seconds: elapsed, qa: prep.qa.length }));
