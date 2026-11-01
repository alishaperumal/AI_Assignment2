// Local PowerPoint authoring. Reads existing artifacts only; never trains or uploads.
// Run with the bundled Node runtime after setting ARTIFACT_NODE_MODULES,
// ARTIFACT_PYTHON and PRESENTATION_SKILL_DIR (see build_person_b_slides.ps1).
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {variantName,selectedMethodDescription,configurationDecision,validationCase,freshMetric} from './presentation_evidence.mjs';

const DELIVERY = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.dirname(DELIVERY);
const BUILD = path.join(DELIVERY, '.build');
const OUT = path.join(DELIVERY, 'output');
const SKILL = process.env.PRESENTATION_SKILL_DIR;
const PYTHON = process.env.ARTIFACT_PYTHON;
const MODULES = process.env.ARTIFACT_NODE_MODULES;
if (!SKILL || !PYTHON || !MODULES) throw new Error('Use build_person_b_slides.ps1 or set the documented runtime variables.');
await fs.mkdir(BUILD, {recursive: true});
await fs.mkdir(OUT, {recursive: true});
try { await fs.symlink(MODULES, path.join(BUILD, 'node_modules'), 'junction'); }
catch (error) { if (error.code !== 'EEXIST') throw error; }
const require = createRequire(path.join(BUILD, 'package.json'));
const { Presentation, PresentationFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL, 'container_tools', 'artifact_tool_utils.mjs')).href);

const readJson = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const exists = async file => !!file && await fs.access(file).then(() => true, () => false);
const absolute = file => path.isAbsolute(file) ? file : path.join(REPO, file);
const content = await readJson(path.join(DELIVERY, 'slide_content.json'));
const references = await readJson(path.join(DELIVERY, 'references.json'));
const sources = await readJson(path.join(DELIVERY, 'result_sources.json'));
const summaryPath = absolute(sources.study_summary);
const study = await exists(summaryPath) ? await readJson(summaryPath) : null;
if (study && study.schema_version !== 1) throw new Error('Unsupported study summary schema.');
const decisionPath = absolute(sources.configuration_decision);
const rawDecision = await exists(decisionPath) ? await readJson(decisionPath) : null;
const decision = configurationDecision(rawDecision,study?.selection?.selected_variant);
const selectedVariant = study?.selection?.selected_variant || rawDecision?.selected_variant;
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const outputArg = process.argv.indexOf('--output');
const readyResults = freshMetric(study?.fresh_seed_summary,'PQ_dataset') && study?.confirmation?.some(r=>r.status==='completed');
const fileName = outputArg >= 0 ? process.argv[outputArg + 1] : `person_b_${readyResults ? 'results' : 'preparation'}_${stamp}.pptx`;
if (!fileName || path.basename(fileName) !== fileName || !fileName.endsWith('.pptx')) {
  throw new Error('--output must be a new .pptx filename without directory components.');
}
const finalPath = path.join(OUT, fileName);
const previewDir = path.join(BUILD, path.parse(fileName).name);
await fs.mkdir(previewDir, {recursive: true});
const presentation = Presentation.create({slideSize: {width: 1280, height: 720}});
const FONT = 'Arial';
const INK = '#142C45';
const MUTED = '#4C5B66';
const ACCENT = '#176C83';
const finite = value => typeof value === 'number' && Number.isFinite(value);
const fmt = (value, digits=3) => finite(value) ? value.toFixed(digits) : 'Pending';
const label = variantName;
const tableSlides = [];
const notes = [];
const representativeRow=study?.fresh_seed_runs?.find(r=>r.seed===study?.representative?.seed);
const baselineRow=study?.confirmation?.find(r=>r.variant==='baseline');

function text(slide, value, x, y, w, h, size=30, color=INK, bold=false) {
  const box = slide.shapes.add({geometry: 'textbox', position: {left:x, top:y, width:w, height:h},
    fill: 'none', line: {fill: 'none', width: 0}});
  box.text = value;
  box.text.insets = 0;
  box.text.style = {typeface: FONT, fontSize: size, color, bold};
  return box;
}

function table(slide, rows, {top=185, height=360, widths=[330,806], size=29}={}) {
  const t = slide.tables.add({rows: rows.length, columns: rows[0].length,
    left:72, top, width:1136, height, columnWidths: widths, values:rows});
  t.borders.assign({style:'solid', fill:'#D8E1E5', width:1});
  t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:rows[0].length}).assign({
    fill:'#FFFFFF', textStyle:{typeface:FONT,fontSize:size,color:INK},
    margins:{left:14,right:14,top:10,bottom:10}, anchor:'center'});
  for(let r=0;r<rows.length;r++) {
    t.rows[r].height = height/rows.length;
    t.getCell(r,0).fill = '#EEF4F6';
    t.getCell(r,0).text.style = {typeface:FONT,fontSize:size,color:INK,bold:true};
  }
  return t;
}

async function picture(slide, file, x=72,y=180,w=1136,h=405) {
  const full = absolute(file);
  const bytes = await fs.readFile(full);
  slide.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),
    contentType:full.toLowerCase().endsWith('.jpg')?'image/jpeg':'image/png',
    alt:'Actual project prediction evidence',fit:'contain',position:{left:x,top:y,width:w,height:h}});
}

function completedRows() {
  const rows = (study?.confirmation || []).filter(r => r.status === 'completed' && finite(r.PQ_dataset));
  if(rows.length > 3) throw new Error('This layout fits the baseline and two full-budget candidates.');
  return rows;
}

function citationNotes(spec) {
  const refs = (spec.references||[]).map(key => `${references[key].short}\n${references[key].url}`).join('\n\n');
  const local = (spec.local_sources||[]).map(file=>absolute(file)).join('\n');
  return `\n\nSources\n${refs}${local ? '\n\nProject evidence\n'+local : ''}`;
}

for(const [index,spec] of content.slides.entries()) {
  const slide = presentation.slides.add();
  slide.background.fill = '#FFFFFF';
  text(slide,spec.title,72,46,910,76,46,INK,true);
  text(slide,sources.speaker,72,674,920,26,23,MUTED);
  text(slide,String(index+1),1160,674,48,26,20,MUTED);
  let narration = spec.speaker_notes;
  const evidence = [];
  if(['baseline','training','experiments'].includes(spec.kind)) {
    if(spec.subtitle) text(slide,spec.subtitle,72,126,910,40,29,ACCENT);
    table(slide,spec.rows,{top:spec.subtitle?191:182,height:spec.rows.length===4?344:370});
    tableSlides.push(index+1);
    const takeaway = spec.kind==='training' && selectedVariant==='no_augmentation'
      ? 'Selected setting: added transforms off. Random foreground-biased crops remain.' : spec.takeaway;
    text(slide,takeaway,72,580,1136,72,28,ACCENT);
    if(spec.kind==='training' && selectedVariant) {
      narration += ' '+selectedMethodDescription(selectedVariant);
      evidence.push(`Frozen selected configuration: ${selectedVariant}\n${decisionPath}`);
    }
    if(spec.kind==='experiments' && study?.protocol) {
      evidence.push(`Study protocol: ${JSON.stringify(study.protocol)}`);
    }
  } else if(spec.kind==='results') {
    const complete = completedRows();
    if(complete.length) {
      const rows = [['Full-budget run','Seed','Epochs','PQ','SQ','RQ'], ...complete.map(r=>[
        label(r.variant),String(r.seed),String(r.epochs_completed),fmt(r.PQ_dataset,6),fmt(r.SQ_dataset),fmt(r.RQ_dataset)])];
      table(slide,rows,{top:170,height:54*rows.length,widths:[326,105,135,190,190,190],size:27});
      tableSlides.push(index+1);
      const fresh=study.fresh_seed_summary;
      const pq=freshMetric(fresh,'PQ_dataset');
      if(decision) text(slide,decision.slide,72,400,1136,46,25,ACCENT);
      if(pq) {
        text(slide,`Fresh-seed PQ: ${fmt(pq.mean,5)} ± ${fmt(pq.std_across_runs,5)}`,72,455,1136,48,34,ACCENT,true);
        text(slide,`${fresh.n_runs} runs, seeds ${fresh.seeds.join(', ')}. Sample standard deviation across runs.`,72,506,1136,36,24,MUTED);
        narration=`The full-budget selection runs were a near-tie. We chose the numerical winner, with added transforms off. Across three fresh seeds, pooled PQ averaged ${fmt(pq.mean,5)}, with sample standard deviation ${fmt(pq.std_across_runs,5)}. This describes training variability, not proof of superiority. The displayed seed ${representativeRow.seed} checkpoint is the best fresh run, not the average. It missed more reference instances than baseline: ${representativeRow.fn} versus ${baselineRow.fn}. But it produced fewer extras: ${representativeRow.fp} versus ${baselineRow.fp}. Higher PQ therefore came with a tradeoff. We have no Kaggle score.`;
        const dice=freshMetric(fresh,'pixel_dice_mean'), iou=freshMetric(fresh,'pixel_iou_mean');
        if(dice && iou) {
          text(slide,`Foreground Dice ${fmt(dice.mean)} ± ${fmt(dice.std_across_runs)}     IoU ${fmt(iou.mean)} ± ${fmt(iou.std_across_runs)}`,72,553,1136,40,25,MUTED);
          evidence.push(`Foreground Dice ${fmt(dice.mean)} ± ${fmt(dice.std_across_runs)} and IoU ${fmt(iou.mean)} ± ${fmt(iou.std_across_runs)} are fresh-seed means/sample SDs. Each run first averages semantic foreground-mask scores over annotation sets. These differ from pooled instance PQ and best-per-filament overlap. Good pixel coverage can still merge or split objects.`);
        }
      } else {
        text(slide,'Fresh-seed assessment is still pending',72,520,1136,60,32,ACCENT);
        narration=(decision?.narration || 'The table contains completed full-budget selection runs.')+' The fresh-seed assessment is still pending. No final mean, standard deviation or representative checkpoint is inferred. This work has not uploaded a submission.';
      }
      evidence.push(`Pooled counts compare predictions against every available annotation set; they are not unique-image instance totals. Seed 42 selects the configuration and is excluded from the fresh-seed mean. Representative seed ${representativeRow.seed} has PQ ${representativeRow.PQ_dataset}; it is the highest-scoring fresh run, not the mean run. Best-per-reference-filament Dice is ${representativeRow.filament_dice_mean} versus ${baselineRow.filament_dice_mean} for baseline.\nSummary: ${summaryPath}\nConfiguration decision: ${decisionPath}`);
      evidence.push('Plain-language metric guide: PQ combines finding the right objects with tracing their shapes. SQ measures overlap quality for matched objects. RQ penalizes missed and extra objects. Dice and IoU measure how much two foreground masks overlap; they do not by themselves check whether separate filaments stay separate.');
    } else {
      text(slide,spec.pending_text,72,225,1136,130,42,ACCENT,true);
      text(slide,'The slide will use completed full-budget comparisons and separate fresh-seed statistics.',72,385,1080,105,32,MUTED);
      narration=(decision?.narration || 'The modelling pipeline and study design are ready.')+' The final study report, fresh-seed mean and representative-model evidence are still pending in this preparation deck. No statistic is inferred from unfinished training.';
    }
    text(slide,spec.takeaway,72,615,1136,45,25,MUTED);
  } else if(spec.kind==='examples') {
    const example=spec.case_role==='test'
      ? (study?.examples?.test||[]).find(e=>e.unlabeled===true && e.n_predicted_instances>0 && e.reason && (e.panel || e.panel_path))
      : validationCase(study?.examples?.validation,spec.case_role);
    const panel=example?.zoom_panel || example?.panel || example?.panel_path;
    if(panel && await exists(absolute(panel))) {
      if(spec.case_role!=='test') text(slide,'Red: human outlines. Cyan: predictions. Representative checkpoint: seed 45.',72,126,1136,40,25,ACCENT);
      await picture(slide,panel,72,185,1136,395);
      if(spec.case_role!=='test') {
        for(const [j,heading] of ['Whole Sun','Human reference','Baseline','Selected model'].entries()) {
          text(slide,heading,95+j*282,187,248,36,25,INK,true);
        }
      }
      if(spec.case_role==='test') {
        text(slide,'Unlabelled test illustration. No ground-truth accuracy ranking.',72,592,1136,64,27,ACCENT);
        narration=`This is an unlabelled test image, selected by typical prediction count for its instrument. The exact rule is: ${example.reason}. The panel shows the final probability map and detected regions. We cannot say how accurate these are without labels. The complete export covers 180 test images, with 1831 predicted instances. Three images have no detections; those are recorded in the sidecar. Nothing has been uploaded to Kaggle.`;
      } else {
        const role=spec.case_role==='highest'?'Highest':'Lowest';
        text(slide,`Whole-observation mean PQ: baseline ${fmt(example.baseline_PQ)} → selected ${fmt(example.final_PQ)}.`,72,592,1136,64,27,ACCENT);
        narration=spec.case_role==='highest'
          ? `This is the highest-scoring validation observation for our representative checkpoint. Both models trace the same filament fairly similarly. Across the whole image, the baseline also has three unmatched extra predictions, while the selected model has none. That helps explain the score increase from ${fmt(example.baseline_PQ)} to ${fmt(example.final_PQ)}. The detail window is only for visibility; the score is for the whole observation.`
          : `This is the lowest-scoring validation observation. The selected model covers much less of the long human-marked filament and adds a tiny island. None of its predicted objects reaches the required matching overlap, so PQ is zero, compared with ${fmt(example.baseline_PQ)} for the baseline. This is a real regression. The detail window is not scored separately.`;
        evidence.push('These are per-observation means, not pooled dataset PQ. Interpret fragmentation/merging alongside recognition quality and TP/FP/FN: missing more filaments can reduce both relation-error counts.');
      }
      evidence.push(`Observation: ${example.observation}\nSelection rule: ${example.reason}\nPanel: ${absolute(panel)}`);
    } else {
      text(slide,spec.pending_text,72,225,1136,140,41,ACCENT,true);
      text(slide,'Inspect missed faint structures, merges, breaks and false islands in the actual outputs.',72,405,1100,110,32,MUTED);
      text(slide,spec.takeaway,72,590,1136,60,27,ACCENT);
      narration=spec.case_role==='test'?'The unlabelled test panel is still pending. It will illustrate predictions without an accuracy ranking.':'The requested highest/lowest selected-model validation case is still pending. No other case is silently substituted based on its position in a list.';
    }
  } else if(spec.kind==='efficiency') {
    const rep=study?.representative;
    if(rep && finite(rep.parameters)) {
      const timing=rep.end_to_end_test;
      const rows=[['Parameters',`${(rep.parameters/1e6).toFixed(2)} million`],
        ['Inference weights',`${fmt(rep.weights_MiB,1)} MiB`],
        ['Training loop',finite(rep.training_seconds)?`${fmt(rep.training_seconds/60,1)} min, excluding validation and setup`:'Training time pending'],
        ['Full training run',finite(rep.subprocess_wall_seconds)?`${fmt(rep.subprocess_wall_seconds/60,1)} min wall time, latest invocation`:'Invocation wall time pending'],
        ['Test image to CSV',finite(timing?.ms_per_image)?`${fmt(timing.ms_per_image,1)} ms/image, without archive saving`:'Timing pending']];
      table(slide,rows,{top:180,height:370});tableSlides.push(index+1);
      text(slide,spec.takeaway,72,580,1136,76,28,ACCENT);
      narration=`The model has ${(rep.parameters/1e6).toFixed(2)} million parameters and ${fmt(rep.weights_MiB,1)} MiB of weights. The training loop took ${fmt(rep.training_seconds/60,1)} minutes; the full invocation took ${fmt(rep.subprocess_wall_seconds/60,1)} minutes including setup, validation and checkpoints. The checked image-to-CSV export averaged ${fmt(timing.ms_per_image,1)} milliseconds per image on this computer. That includes the first image, but excludes model loading and probability-archive saving. Person C receives the weights, probabilities, instance maps and exact configuration. Advanced post-processing and compression comparisons remain their work.`;
      evidence.push(`Representative: ${rep.run_dir}; seed ${rep.seed}.\nTiming: ${timing.seconds} seconds / ${timing.n_images} images; includes read, preprocessing, transfers, model, components, RLE round-trip and complete CSV writing. No warm-up images were removed. Latest training subprocess time is not cumulative resumed-run time.\nSummary: ${summaryPath}\nBenchmark details: ${absolute('results/person_b_study/inference_benchmarks.json')}`);
    } else {
      text(slide,spec.pending_text,72,220,1136,145,41,ACCENT,true);
      text(slide,spec.takeaway,72,415,1136,115,32,MUTED);
      narration='The final model and complete test-pipeline measurements are still pending. The handoff includes probabilities, instance maps, weights and the saved configuration. Advanced separation, post-processing and compression remain Person C\'s experiments.';
    }
  } else if(spec.kind==='references') {
    let y=170;
    for(const key of spec.references) {
      const ref=references[key];
      text(slide,ref.short,72,y,1136,50,27,INK);
      text(slide,ref.url,72,y+37,1136,36,23,ACCENT);
      y+=78;
    }
  } else throw new Error(`Unknown slide kind: ${spec.kind}`);
  const fullNotes=narration+citationNotes(spec)+(evidence.length?'\n\nMeasured evidence\n'+evidence.join('\n\n'):'');
  slide.speakerNotes.text=fullNotes;
  notes.push(`## ${index+1}. ${spec.title}\n\nSuggested time: ${spec.seconds} seconds.\n\n${fullNotes}\n`);
  console.log(`Authored slide ${index+1}: ${spec.title}`);
}

await fs.writeFile(path.join(DELIVERY,'speaker_notes.md'),`# Person B speaker notes\n\nRead the opening paragraph of each spoken slide; Sources and Measured evidence are supporting notes, not a script to read aloud. Allow about 3–4 minutes for slides 1–7. Slide 8 is an optional test illustration; slides 9–10 are the bibliography. Replace the Person B role label with your name when integrating the team deck.\n\n${study?'Evidence source: '+summaryPath:'Preparation draft: result-dependent narration remains pending.'}\n\n`+notes.join('\n'),'utf8');
await fs.writeFile(path.join(DELIVERY,'references.md'),'# Person B references\n\n'+Object.values(references).map(r=>`- ${r.short} [${r.title}](${r.url}). ${r.use}`).join('\n')+'\n','utf8');
const candidatePath=path.join(previewDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
for(const [i,slide] of presentation.slides.items.entries()) {
  const blob=await presentation.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(previewDir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const result=await finalizePresentation({
  workspaceDir:DELIVERY,candidatePath,finalPath,pythonExecutable:PYTHON,
  integrityValidatorPath:path.join(SKILL,'container_tools','inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools','inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...tableSlides.flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:content.slides.length,requiredNativeTableOwnerSlides:tableSlides,
  requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:[FONT]},
  verifyArtifactToolImport:true,receiptPath:path.join(previewDir,'validation.json')});
console.log(JSON.stringify({finalPath,previewDir,study_summary:study?summaryPath:null,slides:content.slides.length,result},null,2));

