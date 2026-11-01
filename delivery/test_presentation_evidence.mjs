// Lightweight evidence-contract checks. Synthetic fixtures here never enter slides.
import test from 'node:test';
import assert from 'node:assert/strict';
import {configurationDecision,validationCase,freshMetric,selectedMethodDescription} from './presentation_evidence.mjs';

test('validation cases follow explicit roles rather than array positions',()=>{
  const gain={observation:'gain',reason:'Largest observed gain (or smallest drop)',panel:'gain.png'};
  const low={observation:'low',reason:'Largest observed drop (or smallest gain)',reasons:['Lowest selected-model observation mean PQ'],panel:'low.png'};
  const high={observation:'high',reasons:['Highest selected-model observation mean PQ'],panel:'high.png'};
  assert.equal(validationCase([gain,low,high],'highest'),high);
  assert.equal(validationCase([high,gain,low],'lowest'),low);
  assert.equal(validationCase([gain],'highest'),null);
});

test('selection wording preserves the actual margin and single-seed limit',()=>{
  const decision={selected_variant:'no_augmentation',selection_seed:42,
    ranking:[{variant:'resolution_1024',PQ_dataset:0.2000},{variant:'no_augmentation',PQ_dataset:0.2006}],
    margin_to_runner_up:0.0006,interpretation:'The small candidate difference is not evidence of statistical superiority.'};
  const text=configurationDecision(decision,'no_augmentation');
  assert.equal(text.nearTie,true);
  assert.match(text.slide,/0\.000600/);
  assert.match(text.narration,/one selection seed does not establish statistical superiority/);
  assert.throws(()=>configurationDecision(decision,'baseline'),/different selected/);
});

test('partial or mismatched seed statistics never become a final estimate',()=>{
  const metric={mean:0.2,std_across_runs:0.01,n:3};
  assert.equal(freshMetric({n_runs:2,seeds:[43,44],PQ_dataset:metric},'PQ_dataset'),null);
  assert.equal(freshMetric({n_runs:3,seeds:[43,44,44],PQ_dataset:metric},'PQ_dataset'),null);
  assert.equal(freshMetric({n_runs:3,seeds:[43,44,45],PQ_dataset:{...metric,n:2}},'PQ_dataset'),null);
  assert.equal(freshMetric({n_runs:3,seeds:[43,44,45],PQ_dataset:metric},'PQ_dataset'),metric);
});

test('selected-method explanation retains stochastic cropping',()=>{
  const description=selectedMethodDescription('no_augmentation');
  assert.match(description,/random foreground-biased 512/);
  assert.match(description,/Random crop sampling still changes between epochs/);
});
