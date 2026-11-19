# Recognition roadmap

## Target and scope

The target is the **dense viral capsid material in perforated virus-like particles**. The desired picking area usually lies inside a surrounding membrane. Capsids are often polygonal, but irregular fragments and partly visible capsids can also be intended targets. A membrane without the desired internal material, a support-film boundary, or a small solid dark object is not sufficient evidence of a target. Do not impose a closed membrane, convex polygon, or complete object as a mandatory rule.

The demonstration annotations are particle-pick circles. Multiple circles sample a single capsid. These are useful positive locations, but they are neither exact capsid boundaries nor independent capsid instances. Unmarked pixels are not automatically negative labels. Reconstructing a filled hull around the circles would create a weak pseudo-label, not expert segmentation ground truth.

## What version 2.2 supplies

The screen application now uses a conservative capsid profile by default. It retains more SAM proposals before checking granular interior texture against the nearby scene, refines supported interiors, and resolves nested candidates after the content check. It preserves the previous generic SAM route as a comparison profile. Prompted segmentation remains available for missed targets. Neither SAM's weights nor a capsid-specific network were trained by these code changes.

This profile addresses an observed failure mode: generic proposal quality can favor large background or envelope regions while a useful inner mask is filtered out. The profile is an engineering improvement to proposal selection, not a validated biological classifier. Its sensitivity depends on screenshot zoom, noise, contrast, crowding, and the selected area. It can omit faint material or fragments and still accept unwanted material.

The workflow is separately testable: semi-auto fills only the clicked contour; auto completes every contour in order using the manual grid convention. Those execution tests cannot establish recognition accuracy. The demonstration comparisons use clean images and report proposed area and coverage of the recoverable positive pick locations. They do not report boundary IoU, pixel precision, or generalization from missing ground truth.

The recommended next step is a supervised capsid-interior segmentation model, compared against the shipped profile on independently held-out micrographs. Keep the floating UI, screen capture, contour review, and click executor; replace the recognition backend and its checkpoint.

## 1. Make the existing SAM model useful with prompts

Start with the supplied original SAM ViT-B checkpoint. Encode each clean image once, retain its embedding, and refine a region with a tight box plus positive and negative points. SAM's official predictor supports these prompts and reuses an image embedding for repeated predictions. For this example, a positive point inside the intended inner region and negative points in the surrounding envelope are reasonable experiments. Their success must be checked visually. [SAM predictor documentation and implementation](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py)

Keep an explicit accept/reject step and save accepted, corrected masks. A mask with a high SAM score can still delineate the wrong structure: `predicted_iou` estimates mask quality, and stability measures consistency under threshold changes. Neither is a probability that an object is a capsid. Area and border filters also cannot teach that biological distinction. [Automatic mask generator implementation](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py)

Record correction effort as well as final mask overlap: number of prompts, time per accepted region, and frequency of switching to manual drawing. An interactive tool can be useful even before unattended recognition is reliable.

## 2. Choose an automatic model through a controlled comparison

| Route | Proposed role | Main tradeoff |
| --- | --- | --- |
| Original SAM with prompts | Immediate annotation and picking workflow | A person still identifies the desired structure. |
| Small 2D U-Net / nnU-Net | First baseline for unattended target-versus-background segmentation | Needs representative, consistently labeled images. |
| Fine-tuned SAM / micro-sam | Retain interactive corrections while learning this domain | More involved training and checkpoint integration. |
| MobileSAM or another smaller encoder | Reduce latency if encoding remains the bottleneck | Speed alone does not establish target recognition accuracy. |

### Supervised 2D segmentation

My first automatic-model experiment would predict the intended region directly with a compact 2D U-Net, using nnU-Net as a strong baseline for training and preprocessing. Begin with two labels: background and capsid interior. Consider a separate membrane/exclusion class if it resolves recurring envelope errors. Use an ignore label for genuinely uncertain or incompletely reviewed regions instead of silently treating them as background. The picker needs an accurate area in which to place points; instance identities become necessary when adjacent capsids need separate contour execution or object-level counting. Connected components can separate disjoint predictions but cannot resolve touching instances on their own.

Optimize for the actual picking outcome: avoid placing points outside the expert target while maintaining useful target coverage. Compare loss weighting and a fixed probability threshold on validation data; do not choose a threshold using the final test images. Polygonality is a soft prior. Convexifying every prediction would incorrectly fill gaps between fragments or include unwanted regions.

nnU-Net is a suitable baseline to evaluate: its official documentation describes supervised biomedical semantic segmentation, support for 2D data, and adaptation of preprocessing and network configuration to a dataset. This is a reason to test it, not evidence that it will outperform fine-tuned SAM on these micrographs. [nnU-Net overview](https://github.com/MIC-DKFZ/nnUNet/blob/master/readme.md)

### Fine-tuning SAM with micro-sam

micro-sam provides custom-data training for interactive, automatic, joint, and semantic segmentation. Choose the training objective to match the intended use. Interactive training alone does not train the separate automatic segmentation decoder described by its current documentation. [Training modes and examples](https://github.com/computational-cell-analytics/micro-sam/blob/main/doc/python_library.md)

For instance-based automatic or joint training, annotate all intended objects in each training image. Sparse instance annotations are suitable for interactive training; semantic training uses class labels. Incomplete labels can teach an automatic model to reject unlabeled true targets. [micro-sam annotation requirements](https://github.com/computational-cell-analytics/micro-sam/blob/main/doc/faq.md)

**Keep model versions separate.** The current micro-sam documentation describes micro-sam2 based on SAM2. Original SAM1 functionality remains in `micro_sam.v1`. The bundled `sam_vit_b_01ec64.pth` is an original SAM checkpoint, so changing to a current micro-sam2 model requires the corresponding architecture and loading code. Its microscopy pretraining is a reason to evaluate it, not evidence that it recognizes this capsid target. Test an appropriate starting model, then fine-tune if needed. [Current official overview, training modes, and FAQ](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html)

## 3. Build a useful pilot dataset

The following is a proposed project plan, not a minimum dataset size guaranteed to work.

1. Start with several dozen independent micrographs and a few hundred target regions if feasible. Include variation in contrast, apparent size, orientation, crowding, acquisition session, and image quality. Expand according to observed validation errors.
2. Use clean micrographs or consistent image exports. Save each mask aligned to its input image, with the same dimensions. Do not train on the green illustration or the complete before/after composite. Keep pixel size, preprocessing, and any screen-to-image transform recorded.
3. Include hard negatives: envelopes without the intended interior, small round structures, membrane edges, debris, empty regions, and confusing texture. Include images with no targets so false positive behavior is measurable.
4. Agree on a short labeling protocol. Have a second person review a subset, particularly uncertain boundaries. Treat unresolved regions consistently instead of forcing arbitrary labels.
5. Split by original micrograph before extracting crops or applying augmentation. Keep related crops and repeated views in the same partition. Where practical, reserve entire acquisition sessions or specimens for testing to assess transfer beyond near-duplicate conditions.
6. Maintain a separate validation set for thresholds and model selection, and an untouched final test set. nnU-Net supports explicit custom splits through `splits_final.json`; use this to enforce grouping. [Custom split documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md)

For each reviewed image, save the clean image, aligned capsid mask or instance labels, an explicit ignore mask where needed, and the original micrograph/acquisition identifier. Keep a separate record of known positive pick locations. Review membrane-only particles, dark debris, support-film edges, overlapping capsids, partial capsids, and truly empty fields as distinct failure cases. The before/after illustrations are useful for establishing this labeling protocol; they should not be mistaken for a fully labeled training set.

## 4. Evaluate the actual picking outcome

Compare every model on the same held-out images, the same target definition, and the same sampling settings. Proposed measurements are:

- **Mask overlap:** IoU and Dice against expert masks, reporting distributions across micrographs rather than one favorable example.
- **Object performance:** target recall, false positive regions per micrograph, and failures on touching or partial targets.
- **Pick precision:** the fraction of generated picking points that fall inside the expert target mask. Report “not applicable” when no points are generated rather than assigning perfect precision.
- **Coverage:** missed target area and missed objects, alongside precision. Producing almost no points must not appear successful.
- **Execution quality:** duplicate point count, out-of-region points, and observed missed clicks during actual cryoSPARC playback.
- **Usability and latency:** correction time; cold model loading, image encoding, subsequent prompt prediction, full automatic proposals, and click playback measured separately on the user's hardware.

Choose acceptable error rates for the experiment before comparing models. Select thresholds on validation data, freeze them, and then evaluate the test set. Add difficult corrected examples in later training rounds without moving test examples into training.

## 5. Address speed without assuming improved recognition

An 8-by-8 proposal grid evaluates 64 prompt locations; 16-by-16 evaluates 256. This reduces prompt locations by a factor of four, but it does not establish a fourfold overall speed improvement because encoding and other work remain. Coarser proposals can miss small targets. Measure the tradeoff.

MobileSAM replaces the heavy image encoder with a smaller one and retains a prompt-based pipeline. It is a candidate if cold encoding remains slow, particularly on CPU, but it needs its own model and checkpoint. Benchmark it on the same images; published repository timings do not predict this workstation's performance. [Official MobileSAM repository](https://github.com/ChaoningZhang/MobileSAM)

Prefer the model and workflow that reduce correction time while meeting the required picking precision. A newer or larger generic segmentation model is not sufficient evidence of better recognition for this target.

## 6. Integrate a trained model without changing the screen workflow

Export a versioned inference package containing the model architecture, weights, class meanings, normalization, expected pixel scale or scale augmentation, and decision threshold. Keep the current screenshot-coordinate convention: resize or tile only inside the recognizer, then map the mask back to the exact captured ROI before displaying contours or sampling points. Test image-edge targets and coordinate round-trips explicitly. Never infer microscope calibration from screen pixels.

Load the trained model once in the existing serial worker. Bound tile and batch sizes, check cancellation between batches, and discard stale results using the current task generation. A compact model may reduce CPU inference cost, but measure cold loading, image inference, and click playback separately before claiming a speed gain. The installer should include the chosen compatible runtime and checkpoint with a verified hash; changing a checkpoint filename alone does not change the architecture correctly.
