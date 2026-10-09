import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const build = path.join(root, '.defense-build-20261009');
const modules = 'C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { PresentationFile, FileBlob } = await import(pathToFileURL(path.join(modules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const deck = await PresentationFile.importPptx(await FileBlob.load(path.join(build, 'source_v18_snapshot.pptx')));
const output = path.join(build, 'source_v18_current');
await fs.mkdir(output, { recursive: true });
await fs.writeFile(path.join(output, 'content.ndjson'), (await deck.inspect({ select: '$.slides[*]', include: ['id', 'index', 'title', 'elements.id', 'elements.kind', 'elements.text', 'elements.position', 'notes.text'] })).ndjson);
for (let i = 0; i < deck.slides.items.length; i++) {
  const slide = deck.slides.items[i];
  await fs.writeFile(path.join(output, `slide-${i + 1}.png`), Buffer.from(await (await deck.export({ slide, format: 'png', scale: 1 })).arrayBuffer()));
  await fs.writeFile(path.join(output, `slide-${i + 1}.layout.json`), await (await slide.export({ format: 'layout' })).text());
}
await fs.writeFile(path.join(output, 'montage.png'), Buffer.from(await (await deck.export({ format: 'png', montage: true })).arrayBuffer()));
console.log(JSON.stringify({ output, slides: deck.slides.items.length, masters: deck.masters.items.length }));
