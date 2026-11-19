# SAM validation on the supplied example

## What this test establishes

The revised recognition code loads the supplied SAM ViT-B checkpoint and completes real inference. Sparse automatic proposals were faster on this example, and cached prompts were much faster after the first image encoding. **Neither automatic configuration correctly identified the three intended inner regions.** Guided prompts and visual selection of alternatives produced closer, still imperfect masks. Total process memory remained approximately 3.1 GiB.

This is a diagnostic on one image, not a trained-model evaluation or a claim that automatic recognition is solved.

## Test setup

The input was the 640 × 454 pixel unannotated left panel cropped from the user's example, included as `examples/example_input.png`. The supplied `sam_vit_b_01ec64.pth` was used unchanged. It loaded with `torch.load(weights_only=True, map_location="cpu")`, matching the ViT-B architecture under strict state loading.

Dependencies were installed into an isolated `validation/env` environment:

- Python 3.12.
- PyTorch 2.5.1+cpu and torchvision 0.20.1+cpu.
- Official `facebookresearch/segment-anything` commit `dca509fe793f601edb92606367a655c15ac00fdf`.
- CPU execution with four PyTorch threads and inference mode enabled.

The original automatic configuration used 16 × 16 prompt positions, SAM's default batch size of 64, predicted-IoU threshold 0.88, stability threshold 0.90, and box NMS threshold 0.4. Its original area and image-edge filters were then applied. The original timing covers mask generation; its lightweight subsequent filtering is excluded.

The revised configuration used 8 × 8 prompt positions, batches of eight, predicted-IoU threshold 0.80, stability threshold 0.90, box NMS threshold 0.7, no additional crop layers, and the revised geometric filtering. Its timing includes that filtering. Neither timed automatic stage includes checkpoint loading, desktop clicking, screenshot capture, or figure rendering.

## Observed timings

| Operation | Time |
|---|---:|
| Initial model and checkpoint loading | 1.44 s |
| Original automatic configuration | 29.22 s |
| Revised sparse automatic configuration | 13.54 s |
| Revised automatic configuration, separate process | 13.13 s |
| First box prompt, including image encoding | 9.37 s |
| Image encoding within that first prompt | 9.27 s |
| Subsequent box prompts using cached encoding | 0.058–0.062 s |
| Subsequent point-and-box refinements using cached encoding | 0.067–0.076 s |

The initial automatic comparison was about 2.16 times faster with revised settings. These are individual measurements on this CPU; they do not establish a general speedup across devices or image collections. Loading time varied between processes, so the initial loading number should not be treated as a guarantee.

## Recognition findings

The original configuration returned ten masks, of which nine passed its geometric filters. The revised configuration returned five masks, all retained. Both selected outer envelopes, the small round object that the reference leaves unselected, and/or edge artifacts. Reducing prompt count improved latency but did not teach SAM the intended target category.

Box prompting produced alternatives closer to the textured inner regions. The assistant then specified boxes and positive/negative points by inspecting the raw input and visually selected the third alternative for each target. These reviewed masks are the guided panel in `validation_results/sam_diagnostic_comparison.png`. They are actual SAM outputs, not generated illustrations or manually painted target masks. Some disconnected fragments and imperfect boundaries remain.

The alternatives most closely matching the intended inner regions were not necessarily those with the highest predicted-IoU score. The unwanted small round object received a predicted-IoU estimate of 1.005 in the original automatic run. This model output is a mask-quality estimate, not a calibrated capsid-class probability or a measured overlap with ground truth.

## Memory and remaining limitations

The combined original/revised process peaked at 3182.3 MiB resident memory. A fresh revised-only process peaked at 3166.6 MiB. **This does not demonstrate a meaningful reduction in total RAM requirements.** Reducing decoder batch size alone did not resolve the model's substantial CPU memory footprint in this test.

No pixel-accurate ground-truth masks were supplied, so Dice, measured IoU, precision, recall, and generalization accuracy are not reported. The reference illustration establishes the intended appearance but is not a validated annotation set. Guided results depend on supplied prompts and visual selection; they must not be described as fully automatic detection. Desktop interaction with Windows/cryoSPARC was outside this inference test.

The 25 recognition regression tests also pass with the installed PyTorch/SAM environment. The included `benchmarks/compare_sam.py` and `benchmarks/refine_example.py` reproduce the tested settings. JSON records and compressed mask arrays retain the measured outputs. `validation_results/summary.json` is the compact machine-readable record.


## Application and distribution validation

The final code passes **82 regression tests**: 25 recognition tests, 15 worker/playback tests, 18 desktop-adapter tests, 21 application-integration tests, and 3 distribution/image-loading tests. These use synthetic masks, fake desktop adapters, and fake model predictors where appropriate. They cover mask holes, overlap deduplication, coordinate transforms, prompt caching, cancellation, pause, stale callbacks/results, wrong-window rejection, image drift, clean exports, and standalone import without starting a GUI.

The generated standalone `main.py` was also run end to end with the actual ViT-B checkpoint and a box/point prompt on the clean example. It produced three mask alternatives and exported 150 grid points for its highest-scored alternative, with no desktop events. That CLI selection is unreviewed and is not a recognition accuracy result.

**Not validated here:** visible Tkinter layout, live Windows capture/click delivery, real cryoSPARC navigation, Windows display scaling, and CUDA execution. The application uses a scrollable control pane, and live automation is deliberately limited to the Windows primary monitor with target/window and screenshot-dimension checks. These guards have fake-backend coverage, not an end-to-end Windows certification.

The code fixes observed failure paths; it cannot establish the exact cause of the user's previous process crashes without the traceback and hardware details. Native model execution can still exhaust RAM/VRAM, and cancellation cannot interrupt an already-running native encoder call.

## Included evidence and rerunning

`validation_results/summary.json` contains the measured summary; detailed settings, prompt coordinates, and scores are in the other JSON files. NPZ files contain actual predicted Boolean masks. The four-panel PNG is a qualitative model-output comparison; the guided panel uses assistant-specified prompts and visually chosen alternatives. It is not a learned automatic target detector.

Run `python benchmarks/compare_sam.py --checkpoint sam_vit_b_01ec64.pth` to repeat the automatic and box-prompt diagnostic, and `python benchmarks/refine_example.py --checkpoint sam_vit_b_01ec64.pth` for the specified corrective prompts. Both scripts write fresh results to `benchmark_output/` and never send desktop events. They use a fixed CPU configuration for comparison; expect different timings on other machines. The checkpoint is not included.
# Version 2.3.0 regression and geometry checks

The 2.3.0 source suite contains 294 passing tests. New cases cover three
successive semi-auto selections with delayed pick-marker rendering, rejection
of actual image changes, complete pick history at result/event handoff,
cancellation at that handoff, GUI-thread release of worker callbacks, removal
of Tcl variable traces, polling recovery and shutdown, retired input surfaces,
application-icon packaging, and native-fault logging. The Windows release gate
also runs three consecutive selections using real Tk events, real overlays and
the real worker, with an in-memory target that cannot send desktop input.

Polygon fitting was replayed over all 12 previously supplied development
examples and their cached SAM proposals. It retained all 47 regions, increasing
the union area from 425,773 to 440,022 pixels (3.35%). It fitted 10 near-convex
polygons and 37 concave polygons. Tiny pinhole cleanup filled 355 pixels;
24,623 pixels in larger holes were retained. The same fitted bitmap drives
outline rendering, hit testing, and point placement. Generic and box-prompt
SAM behavior is unchanged; generic masks matched on all 12 replays.

Coverage of the 2,277 available partial positive picking coordinates changed
from 1,268 to 1,291. These are incomplete annotations on development examples,
not instance masks or exhaustive positive/negative labels. This number is not
precision, recall, IoU, or an independent accuracy estimate. The existing
missed target in example 5 remains missed. Polygon fitting is geometric
regularization and uses the existing weights; it is not a trained capsid model.
Median filter/postprocessing time in the paired replay was 0.233 seconds before
and 0.286 seconds after. SAM encoder/decoder inference was not rerun for this
cached comparison and no inference speedup is claimed.

The reproduced second-selection failure treated late-rendered pick markers
as image movement. Known actual-pick neighborhoods are now excluded from the
alignment guard while the rest of the view is checked. Separately, worker/Tcl
callback lifetime and event-order defects were repaired. Without the user's
original crash log, these checks cannot identify the exact cause of that
particular unexpected process exit. The installed launcher now writes
supported native exception tracebacks to `native_fault.log`.

Manual sampling and the desktop adapter remain byte-identical to 2.2.0, as
does the click-playback implementation. Manual gesture handlers and the manual
result branch are unchanged. Shared changes concern ownership, polling,
shutdown and the requested icon; the control layout is unchanged.

---

