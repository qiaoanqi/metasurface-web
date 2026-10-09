import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const ROOT = process.cwd();
const BUILD = path.join(ROOT, '.defense-build-20261009');
const OUT = path.join(ROOT, 'competition', '答辩材料_20261009');
const SKILL = 'C:/Users/A/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const MODULES = 'C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const PYTHON = 'C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = MODULES;

const { PresentationFile, FileBlob } = await import(pathToFileURL(path.join(MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const sourcePath = path.join(OUT, 'AI超表面结构色智能设计系统_答辩修订版_v21.pptx');
const authoredPath = path.join(BUILD, 'authored_v22.pptx');
const candidatePath = path.join(BUILD, 'candidate_v22.pptx');
const finalPath = path.join(OUT, 'AI超表面结构色智能设计系统_答辩修订版_v22.pptx');
for (const p of [authoredPath, candidatePath, finalPath]) {
  try { await fs.access(p); throw new Error(`Preserve existing file: ${p}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}
const deck = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const slide = deck.slides.items[1];

const layout = JSON.parse(await (await slide.export({ format: 'layout' })).text());
const teamSubtitle = layout.elements.find((e) => e.kind === 'textbox' && e.position.top === 116);
const title = layout.elements.find((e) => e.text === '团队成员与分工');
if (!teamSubtitle || !title) throw new Error('Anonymous slide headings not found');
deck.resolve(teamSubtitle.id).text = '算法、交互与答辩材料';
deck.resolve(title.id).text = '项目实现与分工';

const table = slide.tables.items[0];
if (!table || table.rows.length !== 4) throw new Error('Expected the four-row responsibility table');
const values = [
  ['职责', '负责内容'],
  ['项目研发', '核心算法、系统开发与项目统筹'],
  ['交互设计', '离线与云端系统 UI 设计'],
  ['答辩材料', '大纲整理与 PPT 制作'],
];
for (let r = 0; r < values.length; r++) {
  for (let c = 0; c < values[r].length; c++) table.getCell(r, c).value = values[r][c];
}
const meta = JSON.parse(await fs.readFile(path.join(BUILD, 'slide_content_v21.json'), 'utf8'));
meta[0].script = meta[0].script.replace('我是404 Not Found队的主讲人。我们汇报的作品是', '我们汇报的作品是');
meta[1].title = '项目实现与分工';
meta[1].script = '项目工作按职责划分。项目研发负责核心算法、系统开发与项目统筹。交互设计负责离线与云端系统UI设计，答辩材料负责大纲整理与PPT制作。接下来重点介绍算法如何接入软件，以及现有证据能支持哪些结论。';
meta[1].sources = ['项目职责说明'];
slide.speakerNotes.text = `建议用时 25 秒\n${meta[1].script}\n\n证据来源\n${meta[1].sources.join('\n')}`;
deck.slides.items[0].speakerNotes.text = deck.slides.items[0].speakerNotes.text.replace('我是404 Not Found队的主讲人。我们汇报的作品是', '我们汇报的作品是');

for (const s of deck.slides.items) {
  const l = JSON.parse(await (await s.export({ format: 'layout' })).text());
  for (const e of l.elements) {
    if (e.kind === 'notes') continue;
    if (typeof e.text !== 'string') continue;
    const replacement = e.text
      .replaceAll('404 Not Found队', 'AI 超表面结构色智能设计系统')
      .replaceAll('乔安琪', '')
      .replaceAll('陈雍杰', '')
      .replaceAll('郭千弘', '');
    if (replacement !== e.text) deck.resolve(e.id).text = replacement;
  }
}

await fs.writeFile(path.join(BUILD, 'slide_content_v22.json'), JSON.stringify(meta, null, 2));
await (await PresentationFile.exportPptx(deck)).save(authoredPath);
// Only transplant edited text. Preserve original chart workbooks, fonts and assets.
const merge = spawnSync(PYTHON, [path.join(ROOT, 'competition/merge_defense_anonymization_v22.py'), '--source', sourcePath, '--authored', authoredPath, '--output', candidatePath], { encoding: 'utf8' });
if (merge.status !== 0) throw new Error(merge.stdout + merge.stderr);
console.log(merge.stdout.trim());
await finalizePresentation({
  workspaceDir: ROOT,
  candidatePath,
  finalPath,
  pythonExecutable: PYTHON,
  integrityValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit', '--validate-bullet-geometry', ...[2, 4, 8, 12, 13].flatMap(n => ['--require-native-table-slide', String(n)])],
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [2, 4, 8, 12, 13],
  requiredNativeChartOwnerSlides: [7, 9],
  materializeLiteralChartWorkbooks: false,
  fontPolicy: { basis: 'reference', families: ['Microsoft YaHei', '微软雅黑', '方正公文小标宋'], referencePath: sourcePath, referenceSha256: createHash('sha256').update(await fs.readFile(sourcePath)).digest('hex') },
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD, 'validation_v22.json'),
});

const finalDeck = await PresentationFile.importPptx(await FileBlob.load(finalPath));
const rendered = path.join(BUILD, 'final_slides_v22');
await fs.mkdir(rendered, { recursive: true });
for (let i = 0; i < finalDeck.slides.items.length; i++) {
  const s = finalDeck.slides.items[i];
  await fs.writeFile(path.join(rendered, `slide-${i + 1}.png`), Buffer.from(await (await finalDeck.export({ slide: s, format: 'png', scale: 1 })).arrayBuffer()));
  await fs.writeFile(path.join(rendered, `slide-${i + 1}.layout.json`), await (await s.export({ format: 'layout' })).text());
}
await fs.writeFile(path.join(BUILD, 'final_montage_v22.png'), Buffer.from(await (await finalDeck.export({ format: 'png', montage: true })).arrayBuffer()));
console.log(JSON.stringify({ finalPath, slides: finalDeck.slides.items.length }));
