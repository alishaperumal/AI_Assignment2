# Person B speaker notes

Read the opening paragraph of each spoken slide; Sources and Measured evidence are supporting notes, not a script to read aloud. Allow about 3–4 minutes for slides 1–7. Slide 8 is an optional test illustration; slides 9–10 are the bibliography. Replace the Person B role label with your name when integrating the team deck.

Evidence source: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\summary.json

## 1. Baseline model

Suggested time: 30 seconds.

I handled the models, training and controlled comparisons. U-Net predicts a filament probability for each pixel, using a ResNet34 encoder initialized from ImageNet. We threshold those probabilities and label connected regions. This gives separate objects, but touching filaments can merge and broken masks can split. Person C can improve that conversion using our saved probabilities.

Sources
Ronneberger, Fischer & Brox (2015). U-Net.
https://arxiv.org/abs/1505.04597

He et al. (2016). Deep Residual Learning for Image Recognition.
https://arxiv.org/abs/1512.03385

segmentation_models_pytorch, version 0.6.0.
https://github.com/qubvel-org/segmentation_models.pytorch

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\models.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\inference.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\configs\baseline.yaml

## 2. Training and validation

Suggested time: 40 seconds.

Filaments cover very little of each image. BCE learns pixel classification, while Dice emphasizes foreground overlap. We use AdamW, cosine scheduling and mixed precision. All labels for one observation stay together. Validation scores complete images and pools matching counts across annotation sets, following the organizer's reference. A match requires IoU above 0.5. The selected U-Net/ResNet34 keeps BCE + Dice and random foreground-biased 512 × 512 crops. It disables the added flips, rotations, affine and appearance transforms. Random crop sampling still changes between epochs.

Sources
Milletari, Navab & Ahmadi (2016). V-Net.
https://arxiv.org/abs/1606.04797

Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization.
https://arxiv.org/abs/1711.05101

PyTorch 2.14 documentation. CosineAnnealingLR.
https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

Kirillov et al. (2019). Panoptic Segmentation.
https://arxiv.org/abs/1801.00868

Ahmadzadeh (2026). Self Evaluation Notebook, version 6.
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\losses.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\train.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\data.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\splits\split_v1.json
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\evaluation_reference.json

Measured evidence
Frozen selected configuration: no_augmentation
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\configuration_selection.json

## 3. Controlled comparisons

Suggested time: 35 seconds.

We kept the split fixed and matched sample budgets and planned update attempts. The six-epoch screen can favor quick learners, so the strongest candidates received full-budget runs before selection. Fresh seeds then measured variability. A few AMP overflow skips can change applied update counts. The 1024 experiment also changes field of view, and consensus changes observation weighting, so those comparisons have limits.

Sources
Salehi, Erdogmus & Gholipour (2017). Tversky loss.
https://arxiv.org/abs/1706.05721

Zhou et al. (2018). UNet++.
https://arxiv.org/abs/1807.10165

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\experiments.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\configs\
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\

Measured evidence
Study protocol: {"path":"C:\\Users\\user-pc\\Desktop\\COS711_Assig2\\AI_Assignment2\\results\\person_b_study\\protocol.json","primary_metric":"PQ_dataset","split_sha256":"58245fc1fdc3c39c7e482d703cd4c9e26ebdbcae6e08c6c328149d417284eec3","selection_seed":42,"fresh_seeds":[43,44,45]}

## 4. Validation results

Suggested time: 40 seconds.

The full-budget selection runs were a near-tie. We chose the numerical winner, with added transforms off. Across three fresh seeds, pooled PQ averaged 0.33695, with sample standard deviation 0.00758. This describes training variability, not proof of superiority. The displayed seed 45 checkpoint is the best fresh run, not the average. It missed more reference instances than baseline: 507 versus 449. But it produced fewer extras: 1007 versus 1416. Higher PQ therefore came with a tradeoff. We have no Kaggle score.

Sources
Kirillov et al. (2019). Panoptic Segmentation.
https://arxiv.org/abs/1801.00868

Ahmadzadeh (2026). Self Evaluation Notebook, version 6.
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\summary.json
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\configuration_selection.json

Measured evidence
Foreground Dice 0.640 ± 0.003 and IoU 0.488 ± 0.004 are fresh-seed means/sample SDs. Each run first averages semantic foreground-mask scores over annotation sets. These differ from pooled instance PQ and best-per-filament overlap. Good pixel coverage can still merge or split objects.

Pooled counts compare predictions against every available annotation set; they are not unique-image instance totals. Seed 42 selects the configuration and is excluded from the fresh-seed mean. Representative seed 45 has PQ 0.3448101728178549; it is the highest-scoring fresh run, not the mean run. Best-per-reference-filament Dice is 0.6028158367364901 versus 0.6458612145378748 for baseline.
Summary: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\summary.json
Configuration decision: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\configuration_selection.json

Plain-language metric guide: PQ combines finding the right objects with tracing their shapes. SQ measures overlap quality for matched objects. RQ penalizes missed and extra objects. Dice and IoU measure how much two foreground masks overlap; they do not by themselves check whether separate filaments stay separate.

## 5. Highest validation score

Suggested time: 25 seconds.

This is the highest-scoring validation observation for our representative checkpoint. Both models trace the same filament fairly similarly. Across the whole image, the baseline also has three unmatched extra predictions, while the selected model has none. That helps explain the score increase from 0.306 to 0.771. The detail window is only for visibility; the score is for the whole observation.

Sources
Ahmadzadeh (2026). Self Evaluation Notebook, version 6.
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\report.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\runs\person_b_final\

Measured evidence
These are per-observation means, not pooled dataset PQ. Interpret fragmentation/merging alongside recognition quality and TP/FP/FN: missing more filaments can reduce both relation-error counts.

Observation: 20170612135150Bh
Selection rule: Highest selected-model observation mean PQ
Panel: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\figures\validation_zoom_20170612135150Bh.png

## 6. Lowest validation score

Suggested time: 25 seconds.

This is the lowest-scoring validation observation. The selected model covers much less of the long human-marked filament and adds a tiny island. None of its predicted objects reaches the required matching overlap, so PQ is zero, compared with 0.414 for the baseline. This is a real regression. The detail window is not scored separately.

Sources
Ahmadzadeh (2026). Self Evaluation Notebook, version 6.
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\study_report.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\

Measured evidence
These are per-observation means, not pooled dataset PQ. Interpret fragmentation/merging alongside recognition quality and TP/FP/FN: missing more filaments can reduce both relation-error counts.

Observation: 20110221173814Mh
Selection rule: Lowest selected-model observation mean PQ
Panel: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\figures\validation_zoom_20110221173814Mh.png

## 7. Efficiency and handoff

Suggested time: 35 seconds.

The model has 24.43 million parameters and 93.4 MiB of weights. The training loop took 12.4 minutes; the full invocation took 24.2 minutes including setup, validation and checkpoints. The checked image-to-CSV export averaged 536.3 milliseconds per image on this computer. That includes the first image, but excludes model loading and probability-archive saving. Person C receives the weights, probabilities, instance maps and exact configuration. Advanced post-processing and compression comparisons remain their work.

Sources
segmentation_models_pytorch, version 0.6.0.
https://github.com/qubvel-org/segmentation_models.pytorch

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\train.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\predict.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\runs\person_b_final\

Measured evidence
Representative: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\runs\final\no_augmentation_seed45; seed 45.
Timing: 96.53502770001069 seconds / 180 images; includes read, preprocessing, transfers, model, components, RLE round-trip and complete CSV writing. No warm-up images were removed. Latest training subprocess time is not cumulative resumed-run time.
Summary: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\summary.json
Benchmark details: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\inference_benchmarks.json

## 8. Test image illustration

Suggested time: 0 seconds.

This is an unlabelled test image, selected by typical prediction count for its instrument. The exact rule is: Median predicted-instance count for instrument Lh; visual coverage only, accuracy unknown. The panel shows the final probability map and detected regions. We cannot say how accurate these are without labels. The complete export covers 180 test images, with 1831 predicted instances. Three images have no detections; those are recorded in the sidecar. Nothing has been uploaded to Kaggle.

Sources
Ahmadzadeh (2026). Self Evaluation Notebook, version 6.
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Project evidence
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\study_report.py
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\
C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\runs\person_b_final\

Measured evidence
Observation: 20220523234912Lh
Selection rule: Median predicted-instance count for instrument Lh; visual coverage only, accuracy unknown
Panel: C:\Users\user-pc\Desktop\COS711_Assig2\AI_Assignment2\results\person_b_study\figures\test_20220523234912Lh.png

## 9. Model and loss references

Suggested time: 0 seconds.

Bibliography. No narration is needed.

Sources
Ronneberger, Fischer & Brox (2015). U-Net.
https://arxiv.org/abs/1505.04597

He et al. (2016). Deep Residual Learning for Image Recognition.
https://arxiv.org/abs/1512.03385

Milletari, Navab & Ahmadi (2016). V-Net.
https://arxiv.org/abs/1606.04797

Salehi, Erdogmus & Gholipour (2017). Tversky loss.
https://arxiv.org/abs/1706.05721

Zhou et al. (2018). UNet++.
https://arxiv.org/abs/1807.10165

segmentation_models_pytorch, version 0.6.0.
https://github.com/qubvel-org/segmentation_models.pytorch

## 10. Evaluation and training references

Suggested time: 0 seconds.

Bibliography. The project uses ordinary cosine annealing without SGDR restarts. The MAGFiLO paper describes the dataset; only the supplied competition training labels enter this project.

Sources
Ahmadzadeh et al. (2024). A dataset of manually annotated filaments.
https://doi.org/10.1038/s41597-024-03876-y

Kirillov et al. (2019). Panoptic Segmentation.
https://arxiv.org/abs/1801.00868

Ahmadzadeh (2026). Self Evaluation Notebook, version 6.
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization.
https://arxiv.org/abs/1711.05101

PyTorch 2.14 documentation. CosineAnnealingLR.
https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
