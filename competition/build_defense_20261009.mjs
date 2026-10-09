import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import readline from 'node:readline';
import { createReadStream } from 'node:fs';

const ROOT = process.cwd();
const BUILD = path.join(ROOT, '.defense-build-20261009');
const OUT = path.join(ROOT, 'competition', '答辩材料_20261009');
const SKILL = 'C:/Users/A/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const MODULES = 'C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const PYTHON = 'C:/Users/A/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = MODULES;
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { createCanvas, loadImage } = await import(pathToFileURL(path.join(MODULES,'@napi-rs/canvas/index.js')).href);
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(BUILD, { recursive: true });
await fs.mkdir(OUT, { recursive: true });
const FONT = resolvePresentationFont({ fontFamily: 'Microsoft YaHei' });
const readJSON = async p => JSON.parse(await fs.readFile(path.join(ROOT, p), 'utf8'));
const flow = await readJSON('.state/offline_flow_smart_20261006.json');
const audit = await readJSON('competition/tio2_air_day_audit_20260930.json');
const colorAudit = await readJSON('competition/tio2_air_day_color_audit_20260930.json');
const runtimeAudit = await readJSON('.state/offline_feature_runtime_audit_final_20261006.json');
const appAudit = await readJSON('.state/offline_real_app_flows_final_20261006.json');
const mappingFlow = await readJSON('.state/offline_flow_mapping_single_20261006.json');
const lines = readline.createInterface({ input: createReadStream(path.join(ROOT, 'competition/tio2_air_reference_records_v1.jsonl')), crlfDelay: Infinity });
let record;
for await (const line of lines) { if (line.trim()) { record = JSON.parse(line); break; } }
lines.close();
if (record.index !== 4 || record.polarization !== 'p' || audit.records !== 21088 || flow.candidates.length !== 3) throw new Error('Unexpected evidence identity');
if ([runtimeAudit, appAudit].some(a => a.status !== 'pass' || a.checks.some(c => c.status !== 'pass'))) throw new Error('Audit not passing');
if (mappingFlow.payload.status !== 'available' || mappingFlow.payload.cells.length !== 48) throw new Error('Unexpected mapping evidence');
const C = { bg:'#0C111A', ink:'#F6F8FC', muted:'#ADB8C7', line:'#394252', cyan:'#56D5E9', amber:'#FFD173', green:'#78D5AD', violet:'#BAACFF' };
const deck = Presentation.create({ slideSize:{ width:1280, height:720 } });
const meta=[];
function text(s, v, x, y, w, h, size=25, color=C.ink, bold=false, align='left') {
  const a=s.shapes.add({ geometry:'textbox', position:{left:x,top:y,width:w,height:h}, fill:'none', line:{fill:'none',width:0} });
  a.text=v; a.text.style={typeface:FONT,fontSize:size,color,bold,align,valign:'top',autoFit:'none',lineSpacingMultiple:1.15}; return a;
}
function line(s,x,y,w,color=C.line) { s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{fill:color,width:1}}); }
function swatch(s,hex,x,y,w=72,h=44) { s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:hex,line:{fill:C.muted,width:0.5}}); }
function footer(s,n,src='') { if(src)text(s,src,60,663,1085,22,14,C.muted); text(s,String(n).padStart(2,'0')+' / 15',1150,663,75,22,14,C.muted,false,'right'); }
function slide(title,sub,n,seconds,script,sources=[]) {
  const s=deck.slides.add(); s.background.fill=C.bg;
  text(s,title,60,47,1155,title.includes('\n')?145:62,n===1?54:42,C.ink,true);
  if(sub)text(s,sub,62,116,1150,36,22,C.muted);
  s.speakerNotes.textFrame.setText(`建议用时 ${seconds} 秒\n${script}\n\n证据来源\n${sources.join('\n')}`);
  meta.push({slide:n,title,seconds,script,sources}); return s;
}
async function photo(s,rel,x,y,w,h,crop) {
  let bytes=await fs.readFile(path.join(ROOT,rel));
  const source=await loadImage(bytes);
  let width=source.width,height=source.height;
  if(crop){
    const f=v=>typeof v==='number'?v:parseFloat(v)/100;
    const left=Math.round(width*f(crop.left)),top=Math.round(height*f(crop.top));
    width-=left+Math.round(width*f(crop.right));
    height-=top+Math.round(height*f(crop.bottom));
    const canvas=createCanvas(width,height);
    canvas.getContext('2d').drawImage(source,left,top,width,height,0,0,width,height);
    bytes=canvas.toBuffer('image/png');
  }
  const scale=Math.min(w/width,h/height);width*=scale;height*=scale;
  s.images.add({blob:bytes,contentType:'image/png',position:{left:x+(w-width)/2,top:y+(h-height)/2,width,height},fit:'contain',alt:rel});
}
function table(s, values, x,y,w,h,widths,size=24) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
  t.borders.assign({fill:C.line,width:1,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:C.bg,textStyle:{typeface:FONT,fontSize:size,color:C.ink},margins:{left:16,right:16,top:15,bottom:15},anchor:'center'});
  for(let j=0;j<values[0].length;j++){t.getCell(0,j).fill='#202936';t.getCell(0,j).text.style={typeface:FONT,fontSize:size,bold:true,color:C.cyan};}
  return t;
}

// Reuse the source deck's technical order, with evidence-backed figures and readable spacing.
{
const script='各位评委老师好，我是404 Not Found队的主讲人。我们汇报的作品是AI超表面结构色智能设计系统，参赛方向是AI+软件创新。项目面向微纳光学的早期设计，把目标颜色、候选结构、光谱分析和结果导出放在一个本地可运行的软件中。接下来重点介绍软件如何结合AI，以及我们用什么证据验证它。';
const s=slide('AI 超表面\n结构色智能设计系统','',1,20,script,['competition/02_技术方案.md','output/playwright/main8503-preview-1440.png']);
text(s,'面向微纳光学早期设计的 AI 软件',63,225,1150,42,29,C.cyan);
text(s,'目标颜色  /  候选结构  /  光谱分析  /  结果导出',63,294,1150,50,27,C.muted);
await photo(s,'output/playwright/main8503-preview-1440.png',64,391,1152,188,{left:'23.8%',top:'48.2%',right:'3%',bottom:'34.6%'});
text(s,'404 Not Found队',63,604,404,43,30,C.ink,true);
text(s,'算法创新赛 · AI+软件创新   第 4 答辩室 · 序号 3',491,611,727,37,22,C.muted);
footer(s,1);
}
{
const script='团队成员为乔安琪、陈雍杰和郭千弘。我负责核心算法、系统开发和本次主讲。陈雍杰负责UI设计，郭千弘负责PPT制作。';
const s=slide('团队成员与分工','404 Not Found队',2,25,script,['用户确认的成员与实际贡献','校赛通知附件名单']);
table(s,[['成员','分工'],['乔安琪','项目负责人、核心算法、系统开发与主讲'],['陈雍杰','UI设计'],['郭千弘','PPT制作']],64,205,1152,300,[250,902],25);
footer(s,2);
}
{
const script='结构色来自微纳结构对不同波长光的响应，改变直径、高度和周期，颜色就可能发生变化。早期设计者往往先有一个目标颜色，却不知道从哪组参数开始。只看到一个HEX色值，又无法判断光谱形状和结果来源。项目要解决的是这一段工作流：快速提出候选，让用户看到参数、光谱和色差，并把结果带走继续复核。我们的需求分析来自设计流程与项目实践，当前没有把它包装成大规模用户调研结论。';
const s=slide('背景与用户需求','目标用户：光学教学、研究中的早期方案筛选',3,40,script,['competition/02_技术方案.md']);
text(s,'用户先知道颜色，未必知道结构参数',64,193,1120,48,32,C.ink,true);
const rows=[['输入需求','目标色与材料、入射条件'],['设计判断','D / H / P 候选、光谱与色差'],['继续使用','文件导出与后续物理复核']];
rows.forEach((r,i)=>{const y=285+i*100; text(s,r[0],64,y,210,42,27,[C.amber,C.cyan,C.green][i],true);text(s,r[1],322,y,880,46,29);line(s,64,y+64,1148);});
text(s,'当前定位：快速探索与候选筛选',64,603,1130,38,25,C.muted);
footer(s,3);
}
{
const script='这张图来自已保存的真实离线界面。用户先选结构、材料、偏振和角度，再输入目标色并运行搜索。得到候选后，可以应用到预览页，继续查看光谱或导出数据。预览、逆设计、图案、映射、光谱共五个页面，围绕同一参数上下文工作。这里强调的是可用的软件流程。后面会单独说明候选的质量和参考数据，避免把“按钮运行成功”当作物理精度结论。';
const s=slide('实际软件流程','输入目标色，生成候选，再应用、分析和导出',4,35,script,['.state/ui_previews/offline_smart_grid_verified_20261006.png','.state/offline_real_app_flows_final_20261006.json']);
await photo(s,'.state/ui_previews/offline_smart_grid_verified_20261006.png',64,181,866,467,{left:'25%',top:'7%',right:'2%',bottom:'1%'});
text(s,'五个页面',883,183,330,37,26,C.ink,true);
const pages=[
  ['预览','查看结构与颜色',C.cyan],
  ['逆设计','目标色生成候选',C.amber],
  ['图案','图片转结构参数',C.violet],
  ['映射','查看 D-H 颜色',C.green],
  ['光谱','分析波长响应','#F39EAC'],
];
// Native page-navigation diagram, paired with saved UI crops and mapping data.
s.shapes.add({geometry:'line',position:{left:902,top:252,width:0,height:340},fill:'none',line:{fill:C.line,width:2}});
for (let i=0;i<pages.length;i++){
  const [title,description,color]=pages[i],y=231+i*85;
  s.shapes.add({geometry:'ellipse',position:{left:883,top:y+5,width:38,height:38},fill:C.bg,line:{fill:color,width:1.6}});
  text(s,String(i+1),887,y+10,30,27,19,color,true,'center');
  text(s,title,938,y+2,177,34,27,color,true);
  text(s,description,938,y+42,177,30,20,C.muted);
}
await photo(s,'output/playwright/main8503-preview-1440.png',1124,232,92,64,{left:'24.9%',top:'50.3%',right:'66.2%',bottom:'36.7%'});
await photo(s,'.state/ui_previews/offline_smart_grid_verified_20261006.png',1124,317,92,64,{left:'51.3%',top:'12.6%',right:'42.8%',bottom:'75.1%'});
await photo(s,'.state/ui_previews/offline_pattern_verified_20261006.png',1124,402,92,64,{left:'26.4%',top:'50.4%',right:'56.2%',bottom:'26.2%'});
const mapPreview=s.tables.add({rows:6,columns:8,left:1124,top:487,width:92,height:64,values:Array.from({length:6},()=>Array(8).fill(''))});
mapPreview.borders.assign({fill:C.bg,width:0.2,style:'solid'});
mapPreview.cells.block({row:0,column:0,rowCount:6,columnCount:8}).assign({textStyle:{typeface:FONT,fontSize:1},margins:{left:0,right:0,top:0,bottom:0}});
for(let row=0;row<6;row++)mapPreview.rows[row].height=64/6;
for(const cell of mappingFlow.payload.cells){
  const rgb='#'+cell.rgb.map(v=>Math.round(v*255).toString(16).padStart(2,'0')).join('');
  mapPreview.getCell(5-cell.hi,cell.di).fill=rgb;
}
await photo(s,'output/playwright/main8503-spectrum-1440.png',1124,572,92,64,{left:'26.8%',top:'48.7%',right:'33.4%',bottom:'13.3%'});
s.speakerNotes.textFrame.setText(`建议用时 35 秒\n${script}\n\n证据来源\n.state/ui_previews/offline_smart_grid_verified_20261006.png\n.state/offline_real_app_flows_final_20261006.json\noutput/playwright/main8503-preview-1440.png\n.state/ui_previews/offline_pattern_verified_20261006.png\n.state/offline_flow_mapping_single_20261006.json\noutput/playwright/main8503-spectrum-1440.png\n\n右侧为五页功能索引。缩略图来自已保存界面，映射色格使用2026-10-06的48格解析路线存档，未重新仿真。不同页面示例不作为同一次结果或同一物理路线的对照。`);
footer(s,4,'来源：本地界面存档与应用流程记录');
}
{
const script='系统分为界面、计算路由、颜色表达和结果导出几个部分。路由首先检查结构、材料、衬底以及入射条件，再决定使用注册代理模型、解析近似或腔体TMM。每个结果带有模型版本和适用范围。修改参数后，旧候选和旧分析不能继续冒充新条件下的结果。资源不匹配时，只有独立实现且明确标注的基线路线可以回落，否则就返回不可用。这样把能力边界落实到程序行为，而不是只在文档里写一行提醒。';
const s=slide('系统架构与运行环境','Streamlit 界面连接本地模型、颜色计算和文件导出',5,40,script,['app.py','ui_forward_routes.py','ui_inverse_contracts.py','ui_model_resources.py']);
const steps=[['用户输入','结构与材料\n目标色与入射条件'],['上下文检查','适用范围\n资源与数据形状'],['计算路由','注册代理模型\n解析近似 / FP-TMM'],['结果表达','光谱与颜色\nCSV / JSON / PNG']];
steps.forEach((r,i)=>{const x=65+i*292;text(s,String(i+1).padStart(2,'0'),x,216,260,42,26,C.cyan,true);text(s,r[0],x,278,265,45,29,C.ink,true);text(s,r[1],x,343,265,91,24,C.muted);});
line(s,64,458,1152);
text(s,'结果随上下文保存，参数变化后旧结果失效',64,495,1140,46,30,C.green,true);
text(s,'Python · NumPy · PyTorch · ONNX Runtime CPU\n本地准备好依赖后可断网使用，云端已有部署版本',64,562,1140,75,23,C.muted);
footer(s,5);
}
{
const script='AI部分采用残差多层感知机ResMLP。七个输入包括直径、高度、周期、入射角，以及偏振、材料和衬底编码，输出为380到780纳米的81点反射光谱。典型实现使用256维隐藏层和4个残差块，运行时按实际权重确认。ONNX承担CPU前向推理，PyTorch支持梯度搜索，部分路线对多个注册模型取均值。搜索在预测光谱上寻找候选。现有代理来自历史训练资源，早期求解器与背景口径存在已记录风险，绝对物理精度尚未按修复后的求解器重新验证。新审核参考库与它分开管理。';
const s=slide('核心代理模型','ResMLP 学习几何参数与反射光谱的关系',6,45,script,['torch_model.py:_RCWA_ResMLP','ml_module.py:_get_rcwa_sessions']);
text(s,'7 维输入',65,206,400,56,38,C.cyan,true);
text(s,'D、H、P\n入射角、偏振编码\n材料编码、衬底编码',65,295,430,130,26);
text(s,'残差多层感知机',523,206,670,56,38,C.amber,true);
text(s,'典型实现：256 维隐藏层，4 个残差块\n部分路线按注册模型取均值',523,295,675,90,26);
text(s,'81 点反射光谱',523,418,675,53,36,C.green,true);
text(s,'380–780 nm，间隔 5 nm',523,487,675,40,26,C.muted);
text(s,'ONNX 前向推理与 PyTorch 梯度路线使用对应资源',65,579,1140,42,26,C.muted);
footer(s,6,'历史代理用于流程与候选探索，绝对物理精度尚未按修复后求解器重新验证');
}
{
const script='智能网格采用两阶段搜索。第一阶段在图中尺寸范围生成8乘8乘8网格，排除周期小于1.2倍直径的结构，再按CIEDE2000色差保留三个种子。第二阶段在每个种子附近采用正负8纳米的三档偏移，生成3乘3乘3邻域。最后排序、去重，并对舍入后的参数重新计算颜色。512加81表示过滤前采样位置上限，实际评估还受几何过滤与复算影响。这种方式把搜索范围和排序依据落实到程序中，但没有全局最优保证。';
const s=slide('两阶段智能网格搜索','当前模型范围内的候选筛选',7,50,script,['ml_module.py:smart_grid_search']);
const st=[['粗网格','8 × 8 × 8','先检查几何'],['种子','Top 3','按 ΔE00 排序'],['局部细化','3 × 3 × 3','每轴 ±8 nm'],['候选输出','多样性检查','舍入后复算']];
st.forEach((r,i)=>{const x=64+i*292; text(s,r[0],x,212,262,43,29,C.cyan,true);text(s,r[1],x,293,265,48,31,C.ink,true);text(s,r[2],x,363,267,50,24,C.muted);});
line(s,64,449,1152);
text(s,'D 50–350 nm     H 80–600 nm     P 200–600 nm',64,488,1140,44,27);
text(s,'几何过滤：P ≥ 1.2D',64,555,1140,42,27,C.green,true);
footer(s,7,'默认采样位置上限 512 + 81，最终候选另做复算');
}
{
const script='这组结果来自10月6日保存的真实候选JSON。目标为浅蓝色80C8FF，条件是TiO2、SiO2、单柱、TE和零度。第一名的直径350、高度511.5、周期484.9纳米，预测颜色为6C8895，色差约21.67，和目标仍有明显差距。软件完成了搜索、排序、应用和导出，同时把当前候选质量展示出来。这条案例验证的是工作流，不是高精度颜色匹配。用户还可以比较其他条件，候选在制造之前仍需要统一物理口径复核。';
const s=slide('真实候选案例','TiO₂ / SiO₂，单柱，TE，0°；2026-10-06 保存结果',8,45,script,['.state/offline_flow_smart_20261006.json']);
swatch(s,flow.context.target_hex,64,189,83,51);text(s,'目标色 '+flow.context.target_hex,167,195,570,45,30,C.cyan,true);
const vals=[['排名','D / nm','H / nm','P / nm','预测 sRGB','ΔE00'],...flow.candidates.map(c=>[String(c.rank),c.parameters.d.toFixed(1),c.parameters.h.toFixed(1),c.parameters.p.toFixed(1),c.predicted_hex,c.delta_e2000.toFixed(2)])];
table(s,vals,64,283,1152,250,[104,183,183,183,260,239],27);
text(s,'目标色与当前候选仍有明显差距',64,566,1140,47,31,C.amber,true);
footer(s,8,'来源：offline_flow_smart_20261006.json；色差衡量模型显示颜色与目标色');
}
{
const script='下面是独立的审核参考记录，和上一页代理候选不是同一条结果。记录编号4，几何参数140、281、407纳米，条件是TiO2、SiO2、空气背景、p偏振、零度入射。图中绘出原始81个反射率点，显示颜色约为242D49。这条记录的最大R加T减1误差约为3.67乘10的负13次方。它证明该条数据的守恒和颜色转换能复核，但不代表nG151和Nxy256已经完成所有分辨率的收敛验证，也不代表代理模型在这个记录上有同样的精度。';
const s=slide('审核参考光谱','记录 #4；TiO₂ / SiO₂ / air，p 偏振，0°',9,40,script,['competition/tio2_air_reference_records_v1.jsonl','competition/tio2_air_day_audit_20260930.json']);
const chart=s.charts.add('scatter',{position:{left:62,top:195,width:804,height:424},series:[{name:'反射率 R',xValues:record.wavelength_nm,values:record.R.map(v=>Number(v.toFixed(6))),line:{fill:C.cyan,width:2.5},marker:{symbol:'none'}}],scatterOptions:{style:'line'},hasLegend:false,xAxis:{title:'波长 / nm',min:380,max:780,majorUnit:80,textStyle:{fill:C.muted,fontSize:20},line:{fill:C.line,width:1}},yAxis:{title:'反射率',min:0,max:0.3,majorUnit:0.1,numberFormatCode:'0.0',textStyle:{fill:C.muted,fontSize:20},majorGridlines:{fill:C.line,width:1}},chartFill:C.bg,plotAreaFill:C.bg});
applyPresentationChartFont(chart,{fontFamily:FONT});
text(s,'D / H / P',908,211,300,35,24,C.muted);text(s,'140 / 281 / 407 nm',908,256,302,45,28,C.ink,true);
swatch(s,'#242D49',910,338,82,53);text(s,'#242D49',1008,351,195,38,26);
text(s,'最大 |R+T−1|',908,452,300,36,24,C.muted);text(s,'3.67×10⁻¹³',908,499,300,43,29,C.green,true);
text(s,'nG=151，Nxy=256\n原始 81 点，5 nm 间隔',908,565,305,74,21,C.muted);
footer(s,9,'守恒不等同于 RCWA 收敛；图表采用 81 点，R 保留 6 位小数');
}
{
const script='颜色显示使用统一的色度学转换。反射光谱结合D65光源和CIE1931二度配色函数得到XYZ，再得到色度坐标和sRGB显示颜色，候选用CIEDE2000排序。右侧是真实交互页的CIE图，这里是TE路线的界面示例，不与前面的p偏振参考记录混用。21088条参考数据的独立颜色复算中，最大sRGB分量差约1.47乘10的负8次方。这个数字是存储颜色和复算颜色的数值一致性，不是目标色差，也不是实验精度。显示器、光源与制造后的器件仍可能带来额外差异。';
const s=slide('颜色表达与色度复算','反射光谱结合 D65 与 CIE 1931，生成显示颜色',10,40,script,['color_utils.py','competition/tio2_air_day_color_audit_20260930.json','output/playwright/main8503-spectrum-1440.png']);
const items=[['反射光谱','81 点 R(λ)'],['XYZ 与色度','D65，CIE 1931 2°'],['sRGB 显示','矩阵变换与传递函数'],['候选排序','CIEDE2000 色差']];
items.forEach((r,i)=>{const y=202+i*91;text(s,r[0],64,y,287,42,27,[C.cyan,C.green,C.amber,C.violet][i],true);text(s,r[1],343,y,340,45,24);});
await photo(s,'output/playwright/main8503-spectrum-1440.png',759,185,453,411,{left:'68.5%',top:'49%',right:'1%',bottom:'1%'});
text(s,'真实交互页的 TE 示例',775,610,425,34,20,C.muted);
footer(s,10,'1.47×10⁻⁸ 是 sRGB 分量复算差异，不是 ΔE00 或实验误差');
}
{
const script='图案功能把输入图像映射为结构参数阵列。右侧是保存的本地功能检查截图，使用一个4乘3像素测试图，能够生成输入和映射结果。当前这条路线采用精确注册的单柱解析颜色库，在Lab空间做近邻匹配，和前面神经网络的两阶段搜索独立。它可以导出PNG映射图、CSV参数表和JSON上下文信息。图案颜色没有在此承诺完全复原，也没有通过制造工艺验证，所以我们把输出定位为参数数据，不写成可直接交付电子束曝光的制造版图。';
const s=slide('图案映射与文件导出','独立单柱解析路线，按 Lab 近邻匹配',11,35,script,['.state/ui_previews/offline_light_pattern_verified_20261006.png','ui_pattern_contracts.py','output/playwright/main8503-final-exports-20260824/metadata.json']);
text(s,'输入图像',64,212,350,43,31,C.cyan,true);text(s,'图像缩放后提取像素颜色',64,270,358,81,25);
text(s,'结构参数阵列',64,372,355,44,31,C.green,true);text(s,'D / H / P 对应每个输出像素',64,432,360,85,25);
text(s,'PNG / CSV / JSON',64,562,380,42,27,C.amber,true);
await photo(s,'.state/ui_previews/offline_light_pattern_verified_20261006.png',482,180,727,453,{left:'25%',top:'7%',right:'2%',bottom:'20%'});
footer(s,11,'截图为 4×3 功能测试图；输出参数仍需工艺与物理复核');
}
{
const script='测试分为两层。第一层是参考数据审核，共21088条记录、21083组独立几何，五组重复输出相同，无字段、形状、有限值、几何和守恒检查异常。最大点级守恒误差约1.79乘10的负7次方。第二层是软件运行检查，10月6日的存档包含27项实际运行检查和19项真实应用流程检查，覆盖模型资源、搜索应用、导出、参考查询、主题和分析页面。它们使用实际模型，没有伪造返回值。测试证明已记录版本的软件链路能执行，颜色转换一致，不能外推为全材料、全条件的模型准确率，也不能代替实验。';
const s=slide('数据审核与软件验证','物理数据检查与功能测试分别报告',12,45,script,['competition/tio2_air_day_audit_20260930.json','competition/tio2_air_day_color_audit_20260930.json','.state/offline_feature_runtime_audit_final_20261006.json','.state/offline_real_app_flows_final_20261006.json']);
[['21,088','审核参考记录',C.cyan],['0','无效记录',C.green],['1.79×10⁻⁷','最大点级 |R+T−1|',C.amber]].forEach((r,i)=>{const x=64+i*389;text(s,r[0],x,190,370,73,49,r[2],true);text(s,r[1],x,279,370,40,24,C.muted);});
table(s,[['检查层级','已保存结果','范围'],['数据与颜色','字段、81 点、有限值、守恒及复算通过','TiO₂ / SiO₂ / air，p，0°'],['运行与流程','27 项运行检查；19 项应用流程通过','2026-10-06 离线版本']],64,376,1152,221,[220,540,392],23);
footer(s,12,'来源：本地审计 JSON；分辨率设置不构成收敛声明');
}
{
const script='与常见的脚本工作流和只返回单个预测值的工具相比，本项目把候选、光谱、色度和导出连到同一上下文中。表格比较的是工作流形态，不是没有测过的竞品速度。贡献主要有两方面：第一，把代理预测接入有几何约束、色差排序和候选复算的搜索流程。第二，把模型资源、结果上下文和导出文件绑定起来，避免换了材料或偏振却继续展示旧结果。ResMLP、RCWA和CIEDE2000本身是已有方法，我们不声称发明这些算法。项目创新集中在AI软件的组合实现、可追溯性和使用流程。';
const s=slide('同类工作流对比与软件创新','按工作流能力比较，不宣称未经测试的性能领先',13,45,script,['competition/02_技术方案.md','ui_inverse_contracts.py','ml_module.py','ui_analysis_snapshots.py']);
table(s,[['设计环节','分散脚本','单一预测工具','本系统'],['输入与模型条件','由使用者传递','依赖工具约定','上下文与资源检查'],['结果表达','需额外整合','多为单一结果','多候选、光谱与色度'],['结果复用','手工整理文件','依赖单独实现','导出与上下文绑定']],64,197,1152,302,[218,280,280,374],24);
text(s,'约束搜索与候选复算',64,548,550,44,29,C.cyan,true);
text(s,'模型资源与结果上下文绑定',643,548,571,44,29,C.green,true);
footer(s,13,'对比对象是常见工作流形态；核心算法与工具的已有来源见备注');
}
{
const script='近期最合适的应用是光学课程演示和早期方案筛选。在教学中，学生改变几何参数，观察光谱与颜色如何关联；在研究前期，使用候选表和导出文件减少跨脚本整理。推广计划先提供离线包和示例教程，再做小规模任务试用，记录用户完成时间、失败原因与反馈。结构色标识和微纳图案是后续方向，需要补齐制造约束、统一物理复核与实验样品。我们目前没有用户试点、收入或专利数据，因此只介绍可行路径，不把应用潜力写成已经完成的商业成果。';
const s=slide('应用场景与推广路径','先服务教学和早期筛选，再验证器件应用',14,35,script,['competition/02_技术方案.md','competition/11_本地离线演示说明.md']);
text(s,'光学教学',64,204,470,48,34,C.cyan,true);text(s,'理解参数、光谱与颜色的关系\n用现有参考记录做可复核案例',64,281,511,102,27);
text(s,'早期方案筛选',669,204,547,48,34,C.green,true);text(s,'比较候选并导出结构数据\n交给统一物理或实验验证',669,281,547,102,27);
line(s,64,435,1152);
text(s,'01  离线示例与教程',64,486,353,43,27,C.amber,true);
text(s,'02  小规模任务试用',456,486,353,43,27,C.amber,true);
text(s,'03  工艺与实验闭环',851,486,365,43,27,C.amber,true);
text(s,'后续试用拟记录完成时间、失败原因和用户反馈',64,574,1143,46,26,C.muted);
footer(s,14,'推广步骤为计划，当前没有试点效果或商业收入声明');
}
{
const script='总结来说，项目已把目标色搜索、光谱色度分析、图案映射和导出做成了本地软件，并用现有参考审计与真实功能记录支撑它。当前的不足是模型覆盖和目标色匹配仍有限，双柱ONNX与完整制造验证也没有补齐。下一步按统一口径扩充可审核资源，开展任务试用，再推动物理与实验闭环。我们希望这套软件先成为早期设计者能使用、能检查、能继续复核的工具。谢谢各位老师，欢迎提问。';
const s=slide('总结与下一步','目标颜色、候选结构、光谱色度和导出形成完整软件流程',15,20,script,['competition/02_技术方案.md','competition/14_离线功能核对.md']);
text(s,'已完成',64,210,350,50,34,C.cyan,true);text(s,'本地交互、候选搜索、分析与导出\n参考数据审核与真实功能检查',64,286,1115,95,31);
text(s,'待完善',64,424,350,50,34,C.amber,true);text(s,'模型覆盖与候选匹配\n制造约束、任务试用及实验验证',64,494,1115,91,29);
text(s,'404 Not Found队   谢谢',64,605,1130,43,30,C.ink,true);
footer(s,15);
}

await fs.writeFile(path.join(BUILD,'slide_content.json'),JSON.stringify(meta,null,2));
const finalPath=path.join(OUT,'AI超表面结构色智能设计系统_答辩修订版_v18.pptx');
const candidate=path.join(BUILD,'candidate_v18.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,pythonExecutable:PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--validate-bullet-geometry',...[2,4,8,12,13].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:15,requiredNativeTableOwnerSlides:[2,4,8,12,13],requiredNativeChartOwnerSlides:[9],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation_v18.json')});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const rendered=path.join(BUILD,'final_slides_v18'); await fs.mkdir(rendered,{recursive:true});
for(let i=0;i<finalDeck.slides.items.length;i++){
  const p=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(rendered,`slide-${i+1}.png`),Buffer.from(await p.arrayBuffer()));
}
await fs.writeFile(path.join(BUILD,'final_montage_v18.png'),Buffer.from(await (await finalDeck.export({format:'png',montage:true})).arrayBuffer()));
console.log(JSON.stringify({finalPath,slides:15,totalSeconds:meta.reduce((n,s)=>n+s.seconds,0),font:FONT}));
