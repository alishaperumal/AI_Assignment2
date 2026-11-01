// Small evidence-selection helpers. No rendering, filesystem access or model work.
const finite = value => typeof value === 'number' && Number.isFinite(value);

export const variantName = name => ({baseline:'Baseline', tversky:'Tversky', unetplusplus:'U-Net++',
  resolution_1024:'1024 input', disc_mask:'Disc mask', clahe:'CLAHE',
  local_normalization:'Local normalization', no_augmentation:'No added transforms',
  consensus:'Consensus', lr_low:'Low learning rate', lr_high:'High learning rate'}[name] || name || 'Pending');

export function selectedMethodDescription(variant) {
  return variant === 'no_augmentation'
    ? 'The selected U-Net/ResNet34 keeps BCE + Dice and random foreground-biased 512 × 512 crops. It disables the added flips, rotations, affine and appearance transforms. Random crop sampling still changes between epochs.'
    : `The selected configuration is ${variantName(variant)}. Its exact model and input settings are recorded with the run.`;
}

export function configurationDecision(decision, selectedVariant=null) {
  if (!decision) return null;
  if (selectedVariant && selectedVariant !== decision.selected_variant) {
    throw new Error('Final study and frozen configuration decision name different selected variants.');
  }
  const ranking = [...(decision.ranking || [])].filter(r=>finite(r.PQ_dataset))
    .sort((a,b)=>b.PQ_dataset-a.PQ_dataset);
  const selected = ranking.find(r=>r.variant===decision.selected_variant);
  const runnerUp = ranking.find(r=>r.variant!==decision.selected_variant);
  if (!selected || !runnerUp || selected.PQ_dataset < runnerUp.PQ_dataset) {
    throw new Error('Frozen decision needs a selected numerical winner and a runner-up.');
  }
  const margin = selected.PQ_dataset-runnerUp.PQ_dataset;
  if (finite(decision.margin_to_runner_up) && Math.abs(margin-decision.margin_to_runner_up)>1e-12) {
    throw new Error('Frozen decision margin disagrees with its measured scores.');
  }
  const nearTie = /near[ -]?tie|small candidate difference/i.test(decision.interpretation || '');
  const seed = decision.selection_seed;
  return {selected,runnerUp,margin,seed,nearTie,
    slide: `${nearTie?'Near-tie':'Selection comparison'} at seed ${seed}: ΔPQ ${margin.toFixed(6)}. One seed cannot establish superiority.`,
    narration: `${variantName(selected.variant)} scored ${selected.PQ_dataset.toFixed(6)} pooled validation PQ, compared with ${runnerUp.PQ_dataset.toFixed(6)} for ${variantName(runnerUp.variant)}. The margin is ${margin.toFixed(6)} PQ. ${nearTie?'This is a near-tie. ':''}The numerical selection rule picks the higher score, but one selection seed does not establish statistical superiority.`};
}

export function validationCase(examples, role) {
  const prefixes = {highest:'Highest selected-model observation mean PQ', lowest:'Lowest selected-model observation mean PQ'};
  if (!prefixes[role]) throw new Error(`Unknown validation role: ${role}`);
  return (examples || []).find(example=>{
    const reasons = [...(Array.isArray(example.reasons)?example.reasons:[]),example.reason].filter(Boolean);
    return (example.panel || example.panel_path) && reasons.some(reason=>reason.startsWith(prefixes[role]));
  }) || null;
}

export function freshMetric(fresh, key) {
  if (!fresh || fresh.n_runs<3 || !Array.isArray(fresh.seeds) ||
      fresh.seeds.length!==fresh.n_runs || new Set(fresh.seeds).size!==fresh.n_runs) return null;
  const value=fresh[key];
  return value?.n===fresh.n_runs && finite(value.mean) && finite(value.std_across_runs) ? value : null;
}
