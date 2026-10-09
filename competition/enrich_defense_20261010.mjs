import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createReadStream } from 'node:fs';
import readline from 'node:readline';

const ROOT = process.cwd();
const BUILD = path.join(ROOT, '.defense-build-20261009');
const OUT = path.join(ROOT, 'competition', '答辩材料_20261009');
const SKILL = 'C:/Users/A/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const MODULES = 'C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const PYTHON = 'C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const VERSION = process.argv[2] || 'v21';
if (!/^v\d+$/.test(VERSION)) throw new Error('Expected a version such as v19');
process.env.RUNTIME_NODE_MODULES = MODULES;
const { PresentationFile, FileBlob } = await import(pathToFileURL(path.join(MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { createCanvas, loadImage } = await import(pathToFileURL(path.join(MODULES, '@napi-rs/canvas/index.js')).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const sourcePath = path.join(BUILD, 'source_v18_snapshot.pptx');
const deck = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const FONT = 'Microsoft YaHei';
const C = { bg: '#0C111A', ink: '#F6F8FC', muted: '#ADB8C7', line: '#394252', cyan: '#56D5E9', amber: '#FFD173', green: '#78D5AD', violet: '#BAACFF' };
const flow = JSON.parse(await fs.readFile(path.join(ROOT, '.state/offline_flow_smart_20261006.json'), 'utf8'));
const mapping = JSON.parse(await fs.readFile(path.join(ROOT, '.state/offline_flow_mapping_single_20261006.json'), 'utf8'));
const meta = JSON.parse(await fs.readFile(path.join(BUILD, 'slide_content.json'), 'utf8'));
const records = readline.createInterface({ input: createReadStream(path.join(ROOT, 'competition/tio2_air_reference_records_v1.jsonl')), crlfDelay: Infinity });
let reference;
for await (const value of records) { if (value.trim()) { reference = JSON.parse(value); break; } }
records.close();
if (reference.index !== 4 || reference.R.length !== 81) throw new Error('Reference identity mismatch');
const changes = [];

function txt(s, value, x, y, w, h, size = 25, color = C.ink, bold = false, align = 'left') {
  const a = s.shapes.add({ geometry: 'textbox', position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  a.text = value;
  a.text.style = { typeface: FONT, fontSize: size, color, bold, alignment: align, verticalAlignment: 'top', autoFit: 'none', insets: 0, lineSpacingMultiple: 1.15 };
  return a;
}
function rule(s, x, y, w, color = C.line, width = 1) {
  return s.shapes.add({ geometry: 'line', position: { left: x, top: y, width: w, height: 0 }, fill: 'none', line: { fill: color, width } });
}
function arrow(s, x, y, w, color = C.line) {
  const a = s.shapes.add({ geometry: 'connector', kind: 'straight', position: { left: x, top: y, width: w, height: 0 }, line: { fill: color, width: 2 }, tail: { type: 'triangle', width: 'sm', length: 'sm' } });
  a.sendToBack();
  return a;
}
function swatch(s, hex, x, y, w, h) {
  return s.shapes.add({ geometry: 'rect', position: { left: x, top: y, width: w, height: h }, fill: hex, line: { fill: C.muted, width: 0.5 } });
}
function node(s, value, x, y, w, h, color = C.cyan, size = 25) {
  const a = s.shapes.add({ geometry: 'rect', position: { left: x, top: y, width: w, height: h }, fill: C.bg, line: { fill: color, width: 1.3 } });
  a.text = value;
  a.text.style = { typeface: FONT, fontSize: size, color: C.ink, bold: true, alignment: 'center', verticalAlignment: 'middle', insets: 8, autoFit: 'none' };
  return a;
}
async function cleanBody(n) {
  const s = deck.slides.items[n - 1];
  const layout = JSON.parse(await (await s.export({ format: 'layout' })).text());
  for (const e of layout.elements) {
    if (e.position.top > 160 && e.position.top < 653) deck.delete(e.id);
  }
  changes.push(n);
  return s;
}
async function photo(s, rel, x, y, w, h, crop) {
  let bytes = await fs.readFile(path.join(ROOT, rel));
  const image = await loadImage(bytes);
  let width = image.width, height = image.height;
  if (crop) {
    const left = Math.round(width * crop.left), top = Math.round(height * crop.top);
    width -= left + Math.round(width * crop.right);
    height -= top + Math.round(height * crop.bottom);
    const canvas = createCanvas(width, height);
    canvas.getContext('2d').drawImage(image, left, top, width, height, 0, 0, width, height);
    bytes = canvas.toBuffer('image/png');
  }
  const scale = Math.min(w / width, h / height);
  width *= scale; height *= scale;
  s.images.add({ blob: bytes, contentType: 'image/png', position: { left: x + (w - width) / 2, top: y + (h - height) / 2, width, height }, fit: 'contain', alt: rel });
}
function appendNote(n, note, sources = []) {
  const s = deck.slides.items[n - 1];
  const existing = s.speakerNotes.text;
  if (typeof existing !== 'string' || !existing.includes('建议用时')) throw new Error(`Source notes missing for slide ${n}`);
  s.speakerNotes.text = existing + '\n\n图示口径\n' + note + (sources.length ? '\n' + sources.join('\n') : '');
}

// Keep the user's saved cover, team edits, slide furniture and speaker notes.
{
  const s = await cleanBody(3);
  txt(s, '目标颜色', 64, 192, 280, 43, 30, C.cyan, true);
  swatch(s, flow.context.target_hex, 64, 260, 154, 154);
  txt(s, flow.context.target_hex, 64, 434, 280, 40, 26, C.muted);
  arrow(s, 265, 336, 113, C.cyan);
  txt(s, '未知结构参数', 420, 192, 398, 43, 30, C.amber, true);
  ['D', 'H', 'P'].forEach((v, i) => {
    txt(s, v, 420 + i * 123, 267, 105, 60, 48, C.ink, true);
    txt(s, '?', 420 + i * 123, 343, 105, 57, 45, C.amber);
  });
  txt(s, '直径       高度       周期', 420, 434, 402, 36, 24, C.muted);
  arrow(s, 806, 336, 75, C.amber);
  txt(s, '设计者需要', 932, 192, 280, 43, 30, C.green, true);
  txt(s, '候选参数\n光谱与色差\n可导出的结果', 932, 272, 280, 159, 30);
  rule(s, 64, 506, 1152);
  txt(s, '用户先知道颜色，未必知道结构参数', 64, 545, 1152, 49, 34, C.ink, true);
  txt(s, '快速探索与候选筛选，结果交给后续物理复核', 64, 603, 1152, 37, 25, C.muted);
  appendNote(3, '浅蓝色块来自已存档目标色，仅表示用户输入。D/H/P问号表示未知设计变量，不表示搜索结果。');
}
{
  const s = deck.slides.items[3];
  const t = s.tables.items[0];
  t.position = { left: 1061.67, top: 483.2, width: 91.73, height: 62.4 };
  for (let i = 0; i < 6; i++) t.rows[i].height = 62.4 / 6;
  changes.push(4);
}
{
  const s = await cleanBody(5);
  txt(s, '输入', 64, 186, 230, 36, 25, C.cyan, true);
  txt(s, '检查', 348, 186, 220, 36, 25, C.amber, true);
  txt(s, '按资源选择路线', 680, 186, 268, 36, 25, C.violet, true);
  txt(s, '输出', 1024, 186, 190, 36, 25, C.green, true);
  const input = node(s, '目标色与条件', 64, 319, 231, 76, C.cyan);
  const gate = node(s, '上下文与资源', 349, 319, 240, 76, C.amber);
  const routeA = node(s, '注册代理模型', 686, 259, 253, 55, C.violet, 24);
  const routeB = node(s, '解析近似', 686, 330, 253, 55, C.violet, 24);
  const routeC = node(s, 'FP-TMM', 686, 402, 253, 55, C.violet, 24);
  const output = node(s, '光谱与颜色', 1023, 319, 190, 76, C.green, 24);
  s.shapes.connect(input, gate, { kind: 'straight', fromSide: 'right', toSide: 'left', line: { fill: C.line, width: 2 }, tail: { type: 'triangle' } });
  for (const route of [routeA, routeB, routeC]) {
    s.shapes.connect(gate, route, { kind: 'elbow', fromSide: 'right', toSide: 'left', line: { fill: C.line, width: 1.5 }, tail: { type: 'triangle' } });
    s.shapes.connect(route, output, { kind: 'elbow', fromSide: 'right', toSide: 'left', line: { fill: C.line, width: 1.5 }, tail: { type: 'triangle' } });
  }
  txt(s, '结构 / 材料\n入射条件', 64, 430, 231, 74, 23, C.muted);
  txt(s, '适用范围\n数据形状', 349, 430, 240, 74, 23, C.muted);
  txt(s, 'CSV / JSON / PNG', 984, 430, 231, 40, 21, C.muted);
  rule(s, 64, 525, 1152);
  txt(s, '结果随条件保存，参数变化后旧结果失效', 64, 549, 1152, 46, 30, C.green, true);
  txt(s, 'Python / NumPy / PyTorch / ONNX Runtime CPU', 64, 607, 1152, 35, 23, C.muted);
  appendNote(5, '各计算分支是按条件选择的不同路线，并非三路结果互相替代。资源不匹配且没有独立可用基线时返回不可用。');
}
{
  const s = await cleanBody(6);
  txt(s, '7 维输入', 64, 190, 200, 46, 32, C.cyan, true);
  txt(s, '7', 64, 262, 100, 99, 76, C.cyan, true);
  txt(s, 'D / H / P\n角度、偏振\n材料、衬底', 64, 383, 215, 130, 25);
  txt(s, '典型 ResMLP 结构', 345, 190, 572, 46, 32, C.amber, true);
  const project = node(s, '输入投影\n256 维', 345, 268, 148, 95, C.amber, 23);
  const blocks = [0, 1, 2, 3].map(i => node(s, '残差块\n' + (i + 1), 522 + i * 105, 268, 87, 95, C.amber, 21));
  s.shapes.connect(project, blocks[0], { kind: 'straight', fromSide: 'right', toSide: 'left', line: { fill: C.amber, width: 2 }, tail: { type: 'triangle' } });
  for (let i = 1; i < blocks.length; i++) s.shapes.connect(blocks[i - 1], blocks[i], { kind: 'straight', fromSide: 'right', toSide: 'left', line: { fill: C.amber, width: 2 }, tail: { type: 'triangle' } });
  arrow(s, 292, 315, 35, C.cyan);
  arrow(s, 938, 315, 42, C.green);
  txt(s, '81', 1006, 256, 196, 99, 76, C.green, true);
  txt(s, '反射光谱点', 1006, 372, 208, 42, 27, C.green, true);
  txt(s, '380–780 nm\n5 nm 间隔', 1006, 434, 208, 83, 25, C.muted);
  txt(s, '输出层：Linear(256, 81) + Sigmoid', 345, 388, 620, 39, 24, C.muted);
  txt(s, '单个残差块', 345, 456, 230, 37, 23, C.amber, true);
  txt(s, 'y = ReLU(F(x) + x)', 345, 500, 610, 49, 34, C.ink, true);
  rule(s, 64, 569, 1152);
  txt(s, 'ONNX 用于前向推理，PyTorch 支持梯度搜索', 64, 602, 1152, 37, 26, C.muted);
  appendNote(6, '图示依据torch_model.py默认实现：7维输入、256维隐藏层、4个残差块、81维输出及Sigmoid。实际加载时从权重读取维度和块数。残差块计算为ReLU(F(x)+x)。不展示未验证的预测精度。', ['torch_model.py:_ResBlock', 'torch_model.py:_RCWA_ResMLP']);
}
{
  const s = await cleanBody(7);
  txt(s, '01  粗网格', 64, 188, 368, 41, 29, C.cyan, true);
  txt(s, '8 × 8 × 8', 64, 237, 390, 45, 35, C.ink, true);
  const d = [], h = [];
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { d.push(Number((50 + i * 300 / 7).toFixed(6))); h.push(Number((80 + j * 520 / 7).toFixed(6))); }
  const chart = s.charts.add('scatter', {
    position: { left: 64, top: 304, width: 406, height: 284 },
    series: [{ name: '粗网格 P=600 nm 切片', xValues: d, values: h, marker: { symbol: 'circle', size: 5, fill: C.cyan, line: { fill: C.cyan, width: 0 } }, line: { fill: 'none', width: 0 } }],
    scatterOptions: { style: 'marker' }, hasLegend: false,
    xAxis: { min: 50, max: 350, majorUnit: 100, title: { text: 'D / nm', textStyle: { fill: C.muted, fontSize: 18, typeface: FONT } }, textStyle: { fill: C.muted, fontSize: 18 }, line: { fill: C.line, width: 1 } },
    yAxis: { min: 80, max: 600, majorUnit: 130, title: { text: 'H / nm', textStyle: { fill: C.muted, fontSize: 18, typeface: FONT } }, textStyle: { fill: C.muted, fontSize: 18 }, line: { fill: C.line, width: 1 }, majorGridlines: { fill: C.line, width: 0.5 } },
    chartFill: C.bg, plotAreaFill: C.bg,
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  txt(s, '图示为 P=600 nm 切片，共 64 个位置', 64, 604, 465, 33, 20, C.muted);
  arrow(s, 483, 376, 50, C.cyan);
  txt(s, '02  保留种子', 572, 188, 288, 41, 29, C.amber, true);
  txt(s, 'Top 3', 572, 246, 288, 70, 48, C.amber, true);
  txt(s, '按 ΔE00 排序', 572, 334, 288, 40, 24);
  rule(s, 572, 401, 285);
  txt(s, '03  局部细化', 572, 433, 288, 41, 29, C.green, true);
  txt(s, '3 × 3 × 3', 572, 490, 288, 52, 35);
  txt(s, '各轴 −8 / 0 / +8 nm', 572, 568, 293, 39, 24, C.muted);
  arrow(s, 877, 376, 47, C.green);
  txt(s, '04  输出候选', 962, 188, 254, 41, 29, C.violet, true);
  txt(s, '几何过滤\n色差排序\n去重与复算', 962, 276, 254, 156, 28);
  txt(s, 'P ≥ 1.2D', 962, 474, 254, 46, 28, C.green, true);
  txt(s, 'D 50–350 nm\nH 80–600 nm\nP 200–600 nm', 962, 538, 254, 95, 21, C.muted);
  appendNote(7, '散点图为代码默认粗网格的D-H切片，固定P=600nm，此切片64个位置均满足P≥1.2D。坐标展示保留6位小数。它只说明采样设计，不展示任何模拟色差、种子位置或收敛结果。完整网格有8层周期，512+81是过滤前位置上限。', ['ml_module.py:smart_grid_search']);
}
{
  const s = deck.slides.items[7];
  const layout = JSON.parse(await (await s.export({ format: 'layout' })).text());
  const target = layout.elements.find(e => e.text?.startsWith('目标色 #'));
  const targetRect = layout.elements.find(e => e.kind === 'shape' && e.position.top === 189);
  const warning = layout.elements.find(e => e.text === '目标色与当前候选仍有明显差距');
  if (!target || !targetRect || !warning) throw new Error('Candidate slide anchors missing');
  deck.resolve(targetRect.id).position = { left: 64, top: 183, width: 67, height: 67 };
  deck.resolve(target.id).text = '目标色\n' + flow.context.target_hex;
  deck.resolve(target.id).position = { left: 153, top: 184, width: 244, height: 77 };
  deck.resolve(target.id).text.style = { typeface: FONT, fontSize: 24, color: C.cyan, bold: true };
  swatch(s, flow.candidates[0].predicted_hex, 677, 183, 67, 67);
  txt(s, '首位候选\n' + flow.candidates[0].predicted_hex, 767, 184, 267, 77, 24, C.green, true);
  txt(s, 'ΔE00', 1053, 182, 159, 33, 21, C.muted);
  txt(s, flow.candidates[0].delta_e2000.toFixed(2), 1053, 219, 159, 45, 32, C.amber, true);
  deck.resolve(warning.id).position = { left: 64, top: 566, width: 1140, height: 47 };
  changes.push(8);
}
{
  const s = deck.slides.items[8];
  const layout = JSON.parse(await (await s.export({ format: 'layout' })).text());
  const previous = layout.elements.find(e => e.kind === 'chart');
  deck.delete(previous.id);
  const chart = s.charts.add('scatter', {
    position: previous.position,
    series: [{ name: '反射率 R', xValues: reference.wavelength_nm, values: reference.R.map(v => Number(v.toFixed(6))), line: { fill: C.cyan, width: 2.5 }, marker: { symbol: 'none' } }],
    scatterOptions: { style: 'line' }, hasLegend: false,
    xAxis: { min: 380, max: 780, majorUnit: 80, title: { text: '波长 / nm', textStyle: { typeface: FONT, fill: C.muted, fontSize: 20 } }, textStyle: { fill: C.muted, fontSize: 20 }, line: { fill: C.line, width: 1 } },
    yAxis: { min: 0, max: 0.3, majorUnit: 0.1, numberFormatCode: '0.0', title: { text: '反射率', textStyle: { typeface: FONT, fill: C.muted, fontSize: 20 } }, textStyle: { fill: C.muted, fontSize: 20 }, majorGridlines: { fill: C.line, width: 1 } },
    chartFill: C.bg, plotAreaFill: C.bg,
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  appendNote(9, '本轮将图表从同一原始参考记录重新生成，以修复导入后的数据工作簿关联丢失。81个波长不变，R仍保留6位小数，未重新计算物理光谱。');
  changes.push(9);
}
{
  const s = deck.slides.items[9];
  [268, 359, 450].forEach(y => {
    const a = s.shapes.add({ geometry: 'connector', kind: 'straight', position: { left: 78, top: y, width: 0, height: 27 }, line: { fill: C.line, width: 1.5 }, tail: { type: 'triangle', width: 'sm', length: 'sm' } });
    a.sendToBack();
  });
  changes.push(10);
}
{
  const s = await cleanBody(14);
  txt(s, '光学教学', 64, 190, 537, 45, 32, C.cyan, true);
  txt(s, '看清参数、光谱与颜色的关系', 64, 246, 537, 35, 24);
  await photo(s, 'output/playwright/main8503-spectrum-1024.png', 64, 297, 537, 190, { left: 0.319, top: 0.62, right: 0.296, bottom: 0.04 });
  txt(s, '早期方案筛选', 673, 190, 543, 45, 32, C.green, true);
  txt(s, '比较候选，再导出结构参数', 673, 246, 543, 35, 24);
  await photo(s, '.state/ui_previews/offline_gradient_final_verified_20261006.png', 673, 297, 543, 190, { left: 0.265, top: 0.305, right: 0.035, bottom: 0.05 });
  rule(s, 64, 523, 1152);
  const pathSteps = [['01', '离线示例与教程'], ['02', '小规模任务试用'], ['03', '工艺与实验闭环']];
  pathSteps.forEach((r, i) => {
    const x = 64 + i * 405;
    txt(s, r[0], x, 552, 57, 43, 29, C.amber, true);
    txt(s, r[1], x + 70, 552, 294, 43, 26);
    if (i < 2) arrow(s, x + 346, 572, 36, C.amber);
  });
  txt(s, '后续试用记录完成时间、失败原因和用户反馈', 64, 611, 1152, 34, 24, C.muted);
  appendNote(14, '两图是本地软件存档，展示可用于教学及筛选的操作内容，不是课程试点、用户反馈或实验结果。右侧为独立梯度路线的存档，与第8页网格候选分开。', ['output/playwright/main8503-spectrum-1024.png', '.state/ui_previews/offline_gradient_final_verified_20261006.png']);
}
{
  const s = await cleanBody(15);
  txt(s, '已完成的设计流程', 64, 193, 788, 43, 32, C.cyan, true);
  const names = ['目标颜色', '候选结构', '光谱色度', '文件导出'];
  names.forEach((v, i) => {
    const x = 64 + i * 199;
    txt(s, String(i + 1).padStart(2, '0'), x, 290, 174, 48, 34, [C.cyan, C.amber, C.green, C.violet][i], true);
    txt(s, v, x, 354, 174, 44, 28);
    if (i < 3) arrow(s, x + 146, 314, 32, C.line);
  });
  txt(s, '本地交互与结果导出\n参考审核与真实功能检查', 64, 455, 760, 96, 31);
  s.shapes.add({ geometry: 'line', position: { left: 868, top: 198, width: 0, height: 347 }, fill: 'none', line: { fill: C.line, width: 1 } });
  txt(s, '下一步', 926, 193, 285, 43, 32, C.amber, true);
  txt(s, '模型覆盖\n与候选匹配', 926, 275, 285, 86, 28);
  txt(s, '制造约束\n任务试用与实验', 926, 431, 285, 88, 28, C.muted);
  rule(s, 64, 576, 1152);
  txt(s, '404 Not Found队', 64, 610, 650, 43, 30, C.ink, true);
  txt(s, '谢谢，欢迎提问', 866, 610, 350, 43, 30, C.cyan, true);
}

meta[1].script = '团队成员为乔安琪、陈雍杰和郭千弘。我负责核心算法、系统开发和本次主讲。陈雍杰负责离线与云端系统UI设计，郭千弘负责大纲整理和PPT制作。';
deck.slides.items[1].speakerNotes.textFrame.setText(`建议用时 25 秒\n${meta[1].script}\n\n证据来源\n当前v18文件中的用户修改\n校赛通知附件名单`);
await fs.writeFile(path.join(BUILD, `slide_content_${VERSION}.json`), JSON.stringify(meta, null, 2));
const candidatePath = path.join(BUILD, `candidate_${VERSION}.pptx`);
const finalPath = path.join(OUT, `AI超表面结构色智能设计系统_答辩修订版_${VERSION}.pptx`);
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({
  workspaceDir: ROOT, candidatePath, finalPath, pythonExecutable: PYTHON,
  integrityValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit', '--validate-bullet-geometry', ...[2, 4, 8, 12, 13].flatMap(n => ['--require-native-table-slide', String(n)])],
  explicitTotalSlideCount: 15, requiredNativeTableOwnerSlides: [2, 4, 8, 12, 13], requiredNativeChartOwnerSlides: [7, 9],
  materializeLiteralChartWorkbooks: true, fontPolicy: { basis: 'design', families: [FONT, '微软雅黑', '方正公文小标宋'] },
  verifyArtifactToolImport: true, receiptPath: path.join(BUILD, `validation_${VERSION}.json`),
});
const finalDeck = await PresentationFile.importPptx(await FileBlob.load(finalPath));
const rendered = path.join(BUILD, `final_slides_${VERSION}`);
await fs.mkdir(rendered, { recursive: true });
for (let i = 0; i < finalDeck.slides.items.length; i++) {
  const s = finalDeck.slides.items[i];
  await fs.writeFile(path.join(rendered, `slide-${i + 1}.png`), Buffer.from(await (await finalDeck.export({ slide: s, format: 'png', scale: 1 })).arrayBuffer()));
  await fs.writeFile(path.join(rendered, `slide-${i + 1}.layout.json`), await (await s.export({ format: 'layout' })).text());
}
await fs.writeFile(path.join(BUILD, `final_montage_${VERSION}.png`), Buffer.from(await (await finalDeck.export({ format: 'png', montage: true })).arrayBuffer()));
await fs.writeFile(path.join(BUILD, `visual_changes_${VERSION}.json`), JSON.stringify({ sourcePath, changes, unchanged: [1, 2, 11, 12, 13], preservedUserEdits: true, newSimulation: false }, null, 2));
console.log(JSON.stringify({ finalPath, version: VERSION, changes, slides: 15 }));
