# Experimental learned capsid appearance profile

SARU Picker 2.6 can use a locally trained capsid appearance profile alongside the existing SAM ViT-B checkpoint. The floating panel, manual drawing, contour selection and click executor remain the same. The appearance profile is separate from SAM; installing it does not fine-tune or replace SAM's weights.

## Install and select

Extract the complete Windows download. If the package includes `capsid-profile.json`, keep it beside the setup EXE while installing. Setup copies it into `_internal/models`, beside the bundled SAM checkpoint. Choose **Advanced → Detection profile → capsid-learned** to use the experimental learned detector. The default **capsid** setting retains the earlier texture-based detector even when this file is installed. No Python installation, model conversion or extra download is required.

For a source installation, place `capsid-profile.json` beside the configured SAM checkpoint. A custom checkpoint in another folder needs its profile beside it as well. Advanced settings retain **generic** for unclassified SAM proposals. Manual drawing and user box prompts do not require the learned profile.

The experimental setting requires its profile file. A missing or invalid profile produces an actionable error; select **capsid** to use the earlier texture-based detector. Detection diagnostics record the loaded file's identity and SHA-256.

## What the profile learns

The profile learns combinations of local density, texture at multiple scales, directional structure and surrounding contrast. Its small neural network runs through NumPy and SciPy on the CPU. Learned image features select a bounded set of box-and-point prompts for SAM, after which texture and shape checks refine polygonal interiors. The SAM image is encoded once. Dense granular material is stronger evidence than darkness alone. Polygon fitting remains a geometry step; neither a fixed number of sides nor a completely closed surrounding membrane is required.

Training must use clean, unmarked images as input. Positive picking markers indicate intended locations, while unmarked image regions remain unlabeled. A filled region reconstructed from a lattice of markers is only an approximate extent of those picks, not an expert segmentation boundary or a count of biological capsids.

Positive-unlabeled learning is an appropriate reference for this incomplete-label setting. Topaz demonstrates this approach for cryo-EM particle picking, and non-negative PU risk provides a way to train from positives and unlabeled samples. SARU's optional appearance network is a separate implementation; it is not a Topaz or crYOLO checkpoint:

- [Bepler et al., Nature Methods (2019)](https://www.nature.com/articles/s41592-019-0575-8).
- [Kiryo et al., Non-negative PU learning (2017)](https://arxiv.org/abs/1703.00593).

Selecting only the most confident positives introduces selection bias. The positive prevalence used by PU learning is also an assumption. Consequently, this profile's bounded score must not be described as a calibrated probability that a region is a capsid.

## Validation and practical limits

Split evaluation by original micrograph or acquisition session when those identifiers are available. Keep duplicate and overlapping crops together. Fit normalizers and learned parameters only on training groups. Record training-data, feature-schema and profile hashes so cached predictions cannot be mistaken for results from another model.

Incomplete labels support measurements such as coverage of marked locations, selected area, candidate counts and inspection of additional detections. They do not establish pixel precision, full capsid recall, exact boundary IoU or whether every additional detection is wrong. Performance measured during iterative development is not an untouched external validation set.

The learned option is experimental: development evaluation found higher coverage of marked regions, more proposed contours, missed known targets on some images, and unwanted background or structural regions. These findings do not justify replacing the default detector. Review contours in semi-auto mode on representative new micrographs before using automatic picking. Screen zoom, image contrast, ice conditions, magnification and target appearance can differ from the training examples. Preserve high-confidence missed targets and explicitly rejected membrane/background regions for the next training round. The learned path accepts at most 4,194,304 input pixels and rejects a larger screen selection before feature allocation. Select a smaller region if this limit is reached; coordinates are not silently resized.

## Distribution

The runtime accepts a bounded numeric JSON schema and rejects mismatched dimensions, unsupported feature names and invalid parameter values. It does not deserialize executable model objects. The public source/build contains the inference code and synthetic validation fixtures. Private training images, annotations and trained profiles are delivered separately and are not committed to the public repository.
