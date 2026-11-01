# Person C speaker notes

Narration for the Person C slides (also embedded in the PowerPoint notes). About 2.5 minutes for enhancement and compression, then 30 seconds of conclusions shared by the team.

## Where the errors were

I took the seed-45 probabilities from Person B and first asked which errors cost PQ. Pooled over the 200 validation annotation sets there were 783 matched filaments, 1,007 extra predictions and 507 misses, so extra detections outnumber misses two to one. The histogram shows why: 48% of the unmatched predicted regions are smaller than 320 pixels, while only 4% of matched regions and 5% of human-drawn filaments are that small. Fragmentation was 84 cases against 20 merges.

## Post-processing study

Because the probabilities were saved, I built an offline runner that re-scores them with exactly the same evaluator. It first reproduced Person B's PQ of 0.3448101728 bit for bit. I then declared a grid of 55 settings, changing one thing at a time. Removing small regions and bridging nearby fragments helped most; hysteresis and closing helped less. Changing the pixel cutoff alone did nothing, which tells us the misses are not just faint pixels below 0.5. Splitting regions at confident cores with a watershed made things worse. The error bars are 95 percent bootstrap intervals over images.

## Selected post-processing

The selected rule has two steps. First, fragments separated by at most 16 pixels get the same instance ID, without adding any pixels. Then regions smaller than 320 pixels are removed. PQ rises from 0.345 to 0.403. To check that we did not just fit the validation set, I tuned on one half of the validation time blocks and tested on the other half: the gain held at +0.056 and +0.059. Extra predictions halve and fragmentation drops, at a cost of 17 matched small filaments and a few more merges.

## Best and worst cases

These are the validation images that changed most. In the best case, a tiny partial detection of a real filament was removed: it never reached the IoU of 0.5 needed for a match, so it only counted as an extra object. In the worst case, a small real filament below 320 pixels was removed, which is the price of the size filter. Bridged fragments appear as one colour.

## What did not help

Negative results matter too. Moving the pixel cutoff between 0.3 and 0.7 never beat 0.5. Watershed splitting at confident cores made PQ worse because single filaments have uneven confidence and get cut into pieces, and merges were rare anyway. Flip test-time augmentation lowered PQ at our selected setting and, even after re-tuning the cutoff, gained almost nothing while costing nearly four times the model time. Filling holes and four-connectivity did not help either.

## Compression

For compression I compared every variant on the same validation split, the deployable file size, and the complete test pipeline time from reading the image to writing the CSV, on the same RTX 3070. Storing the weights in half precision halves the file with no measurable loss in PQ. Removing 30 percent of the channels in every layer and fine-tuning for ten epochs kept PQ at 0.403 against 0.403 with half the parameters; removing 50 percent was too destructive for the same short fine-tune (0.365). The distilled ResNet18 student scored 0.401 and the label-only control 0.398; before post-processing the order was reversed, so with one run each, distillation's effect is within run-to-run noise. The network itself takes only about 96 of the 456 milliseconds per image: reading the image, post-processing and checking the encoding dominate, so a smaller network barely changes total time. For CPU-only deployment, ONNX Runtime with INT8 weights is four times smaller and faster than PyTorch, without losing PQ.

## Final pipeline and submission

The final pipeline runs whole-image inference, the selected post-processing, and checked RLE encoding. It produced 1421 filaments on the 180 test images. Model size, training time and end-to-end time are on the slide. The images are unlabelled test predictions, shown as illustrations only. The Kaggle score is added after the upload.

## Conclusions

Our main take-away is that turning probabilities into separate objects mattered as much as the network: post-processing alone added more PQ than the whole training-configuration study. Errors were dominated by small extra detections, and a data-driven size filter plus fragment bridging targeted them. For efficiency, pruning and half-precision storage shrank the model a lot at the same validation PQ, but the network is only a small part of the end-to-end time. Our biggest roadblocks were tuning many settings on one validation split, which we handled with declared grids, split-half and bootstrap checks, and limited GPU time, so compression runs used one seed and aggressive pruning needed more fine-tuning than we had. Next time we would train for instances directly, for example with a boundary or skeleton target, prune gradually, repeat compression with several seeds, and submit to Kaggle early to compare local and leaderboard scores.
