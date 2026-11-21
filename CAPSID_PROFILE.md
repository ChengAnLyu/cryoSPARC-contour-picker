# Experimental learned capsid appearance profile

SARU Picker 2.8 can use a locally trained capsid appearance profile alongside the existing SAM ViT-B checkpoint. The floating panel, manual drawing, contour selection and click executor remain the same. The appearance profile is separate from SAM; installing it does not fine-tune or replace SAM's weights.

## Install and select

Extract the complete Windows download. If the package includes `capsid-profile.json`, keep it beside the setup EXE while installing. Setup copies it into `_internal/models`, beside the bundled SAM checkpoint. Choose **Advanced → Detection profile → capsid-learned** to use the experimental learned detector. The default **capsid** setting retains the earlier texture-based detector even when this file is installed. No Python installation, model conversion or extra download is required.

For a source installation, place `capsid-profile.json` beside the configured SAM checkpoint. A custom checkpoint in another folder needs its profile beside it as well. Advanced settings retain **generic** for unclassified SAM proposals. Manual drawing and user box prompts do not require the learned profile.

The experimental setting requires its profile file. A missing or invalid profile produces an actionable error; select **capsid** to use the earlier texture-based detector. Detection diagnostics record the loaded file's identity and SHA-256.

## Shared carbon-boundary check in 2.6.1

The `capsid` and `capsid-learned` detection modes share an additional support-boundary check based on full-image edge context and candidate geometry. It does not alter the screenshot, erase an entire side of a detected line, or retrain either SAM or the numeric appearance profile. A flat polygon side by itself is not sufficient evidence for rejection. Explicit user prompts and manual contours bypass this automatic check.

## What the profile learns

The profile learns combinations of local density, texture at multiple scales, directional structure and surrounding contrast. Its small neural network runs through NumPy and SciPy on the CPU. Learned image features select a bounded set of box-and-point prompts for SAM, after which texture and shape checks refine polygonal interiors. The SAM image is encoded once. Dense granular material is stronger evidence than darkness alone. Polygon fitting remains a geometry step; neither a fixed number of sides nor a completely closed surrounding membrane is required.

Training must use clean, unmarked images as input. Positive picking markers indicate intended locations, while unmarked image regions remain unlabeled. A filled region reconstructed from a lattice of markers is only an approximate extent of those picks, not an expert segmentation boundary or a count of biological capsids.

Positive-unlabeled learning is an appropriate reference for this incomplete-label setting. Topaz demonstrates this approach for cryo-EM particle picking, and non-negative PU risk provides a way to train from positives and unlabeled samples. SARU's optional appearance network is a separate implementation; it is not a Topaz or crYOLO checkpoint:

- [Bepler et al., Nature Methods (2019)](https://www.nature.com/articles/s41592-019-0575-8).
- [Kiryo et al., Non-negative PU learning (2017)](https://arxiv.org/abs/1703.00593).

Selecting only the most confident positives introduces selection bias. The positive prevalence used by PU learning is also an assumption. Consequently, this profile's bounded score must not be described as a calibrated probability that a region is a capsid.

## Confirmed large positives and blank images in 2.8.0

The private 120-example update adds explicit supervision supplied by the user: the blank images in examples 1–80 and all 40 images in examples 81–120 are confirmed negatives for wanted large-capsid picks. The earlier large-capsid markings are confirmed positives. Small markings and other unmarked pixels in nonblank images are not relabeled as negatives. The combined manifest records the confirmation, exact positive region IDs and source-deck hashes for every image.

`--positive-policy confirmed-large` uses only those explicitly confirmed IDs. `--negative-weight` adds a separate logistic loss on valid pixels from confirmed blank images to the existing non-negative positive-unlabeled objective. Confirmed blanks do not enter the unlabeled pool in this mode. Negative batches give each confirmed blank screenshot equal expected weight; `--positive-sampling case-balanced` does the same for positive screenshots. `--positive-prior` exposes the assumed prevalence in the unlabeled pool and defaults to 0.15. Legacy policies with zero negative weight retain the original sampling and loss behavior.

For this dataset, large marked extents are identified by at least 5,000 annotated screen pixels and a nonempty six-pixel-eroded valid core. This yields 95 usable large regions in 55 images. The training selection is an annotation-based size proxy, distinct from the unchanged 3,500-pixel final runtime contour cutoff. The 43 blank images include debris, support-film edges and small objects that should not trigger wanted picks. Only invalid screenshot margins are excluded from their negative masks.

Ten grouped folds hold out each of the 120 examples once. The earlier 80 examples retain their prior fold membership; the new blank examples are added before looking at model results. All three sample pools, feature normalizers, parameters, row indices and batch-draw evidence are bound to the training split. Candidate selection checks actual production SAM contours on held-out images, balancing false picks on confirmed blanks against coverage of the confirmed large marked cores. The release evidence distinguishes this internal candidate comparison from the final profile fitted on all 120 examples. It does not claim independent external validation or calibrated biological probabilities.

The public repository contains the trainer and inference implementation. The private bundle supplies the trained profile, data provenance and measured results. Only `capsid-learned` loads these weights; the default `capsid` heuristic and SAM checkpoint are unchanged.

## Strong-core prompts and size eligibility in 2.8.0

A clear, strongly scored capsid can be joined to weaker surrounding material in the appearance map. The resulting irregular support component may fail the prompt geometry checks before SAM sees the intended core. The learned detector now adds a compact seed pass at appearance score 0.9 after the original 0.5 prompt pass. Existing prompts keep their geometry, order and priority; additional boxes fill only unused slots within the same 24-prompt budget and near-duplicate boxes are skipped. These scores remain uncalibrated. A seed only asks SAM for a proposal; the final appearance, shape, carbon and minimum-size checks still determine whether a contour is selectable.

With a positive minimum capsid area, the learned refiner fits its final polygon and checks each four-connected foreground component before overlap suppression and the final selection limit. A polygon below the cutoff cannot suppress an eligible larger alternative and then disappear at the later size gate. The check uses final polygon pixels, excluding holes; it does not approximate final area with the earlier support mask. The zero-cutoff refinement path retains its earlier behavior, and the final automatic output gate remains in place. The default capsid heuristic, explicit prompts, manual drawing and picking controls are unchanged.

## Training with additional examples in 2.7.0

The private 2.7.0 update combines examples 1–40 and 41–80. Every clean screenshot contributes unlabeled samples; positive supervision comes only from supplied markings. Unmarked screenshots remain unlabeled. Model metadata records both source-deck hashes, the exact manifest, grouped folds, feature and trainer source hashes, and the actual training cases. The installed profile is fitted using all 80 examples after internal comparison with the previous profile.

The training utility accepts `--positive-policy substantial` for the earlier substantial-cluster policy or `--positive-policy all-marked` to retain usable interiors from small confidently marked groups. Both policies erode the reconstructed marked regions by six pixels before selecting positive samples. `--positive-sampling pixel` retains the original pixel sampling; `--positive-sampling case-balanced` gives each screenshot with usable positives equal expected positive-batch weight. Case balancing reuses scarce pixels more often; it does not create additional independent observations. The full profile and every fold record per-case sampling counts, with separate per-row draw evidence for the private reproducibility archive.

Evaluation holds each example out once and keeps visually overlapping crops together. The original 40 and added 40 are evaluated separately, using training-only feature normalizers. Original micrograph and acquisition-session identities were not supplied, so screenshot grouping is a limited substitute. Candidate selection prioritizes large marked targets and their contour boundaries, checks selected area and actual final contours, and reports small marked regions as a secondary measure. Marked extent is a size proxy, not a physical capsid measurement. The private release guide reports the selected policy, observed improvements and remaining failures. These comparisons are internal development evidence, not an independent accuracy benchmark.

## Compact proposals for attached regions in 2.7.0

A strongly scored capsid can be connected to an irregular extension in the appearance-support image. Rejecting the whole connected region before prompting SAM can then miss a clear target. The learned path can derive compact seeds from this support with a stronger, bounded opening. Each seed must pass the existing strict shape tests before its box and point are sent to SAM. The rough parent is never accepted directly as a pickable region. Final texture, shape, polygon and carbon checks retain their existing criteria.

Previously eligible compact prompts keep their original order and take priority under the 24-prompt limit. Additional seeds fill only unused slots. This correction changes learned automatic prompting; manual drawing, explicit box/point prompting and the default capsid detector retain their existing behavior.

## Validation and practical limits

Split evaluation by original micrograph or acquisition session when those identifiers are available. Keep duplicate and overlapping crops together. Fit normalizers and learned parameters only on training groups. Record training-data, feature-schema and profile hashes so cached predictions cannot be mistaken for results from another model.

Incomplete labels support measurements such as coverage of marked locations, selected area, candidate counts and inspection of additional detections. They do not establish pixel precision, full capsid recall, exact boundary IoU or whether every additional detection is wrong. Performance measured during iterative development is not an untouched external validation set.

The learned option is experimental: development evaluation found higher coverage of marked regions, more proposed contours, missed known targets on some images, and unwanted background or structural regions. These findings do not justify replacing the default detector. Review contours in semi-auto mode on representative new micrographs before using automatic picking. Screen zoom, image contrast, ice conditions, magnification and target appearance can differ from the training examples. Preserve high-confidence missed targets and explicitly rejected membrane/background regions for the next training round. The learned path accepts at most 4,194,304 input pixels and rejects a larger screen selection before feature allocation. Select a smaller region if this limit is reached; coordinates are not silently resized.

## Distribution

The runtime accepts a bounded numeric JSON schema and rejects mismatched dimensions, unsupported feature names and invalid parameter values. It does not deserialize executable model objects. The public source/build contains the inference code and synthetic validation fixtures. Private training images, annotations and trained profiles are delivered separately and are not committed to the public repository.

## Large-only output filter in 2.7.1

The 2.7.0 80-example profile is reused unchanged. Automatic `capsid` and `capsid-learned` output now has a minimum **3,500 foreground pixels per connected component**, after all existing refinement and carbon checks. Holes are excluded from area. Disconnected small components cannot pool area, and rejected components are never restored as a fallback. The limit is adjustable in Advanced settings or with `--min-capsid-area`; 0 disables it. This is an output constraint, not a newly trained model or a claim that every sufficiently large region is a capsid. It can omit true small targets and incomplete visible fragments of large targets. Prompt generation is unchanged so compact seeds can still produce large capsid contours.
