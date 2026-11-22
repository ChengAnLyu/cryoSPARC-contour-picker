# Version 2.8.0 validation scope

This update adds confirmed-negative training and explicit confirmed-large positive selection to the small appearance model. It preserves the live screen workflow and the earlier semi-auto repeated-selection correction. The manual sampler, click executor, overlays, navigation, carbon checks, runtime appearance features, default capsid heuristic and 3,500-screen-pixel output floor retain their existing behavior. Application and installer identifiers are updated.

Two learned-path corrections address failures found during the combined-data comparison. Compact score-0.9 seeds can add tight SAM prompts when a strong core is merged with weaker support at the original 0.5 threshold. Original prompt geometry/order and the 24-prompt bound are retained. With a positive configured minimum area, final polygons are checked for four-connected foreground eligibility before learned overlap suppression, preventing an undersized candidate from hiding an eligible larger alternative. The zero-cutoff refinement path remains available. Tests cover supplemental prompt priority, duplicate control, cancellation, final rather than pre-polygon area, separated components, holes, and the recovered size-ordering failure. Private case-level production traces and complete held-out comparisons bind these changes to the actual source implementation.

The private combined dataset contains 120 examples: 43 explicitly confirmed blank images and 95 usable large positive marked regions across 55 images. Other unmarked pixels and smaller annotations in nonblank images remain unknown. Confirmed blank pixels enter a separate negative pool and logistic loss. They are excluded from the unlabeled pool. Case-balanced negative sampling and optional case-balanced positive sampling are recorded with actual sample indices, draw counts, split membership, source hashes and user-confirmation provenance. Normalizers use training unlabeled rows only. Legacy zero-negative-weight behavior is covered by regression checks.

Development comparison uses ten grouped folds with every screenshot held out once, preserving the old 80-case memberships. Candidate profiles run through actual production prompt generation, SAM decoding, refinement, carbon rejection and the size gate. Verified SAM image embeddings can be reused across candidates without reusing their scores or final contours. The frozen old profile is also evaluated on the 40 newly confirmed blank images. Earlier 80-example baseline masks retain their historical identity. Exact measurements, selected hyperparameters, profile hashes and full-data training evidence are in the private release reports.

Negative examples permit direct false-contour and affected-image counts on confirmed blanks. Positive markers remain incomplete and their reconstructed extents are not exact biological boundaries, so positive-core coverage and overreach proxies do not establish full recall, precision on nonblank images or boundary IoU. Micrograph/acquisition identifiers were not provided; overlap-screened screenshot folds are internal development evidence. The final distributed profile is fitted on all 120 examples and is not itself an untouched test model.

The Windows build retains source, frozen and installed gates for the topmost panel, repeated semi-auto selection, navigation, sidecar installation and actual SAM inference. Public Windows fixtures do not contain private training images or weights. The private profile is separately validated in the source runtime; downloaded Windows reports determine native installation results. The default capsid heuristic and SAM weights are not retrained by this update.

# Version 2.7.2 validation scope

This patch corrects exhausted-proposal handling in semi-auto mode. After a pick completes, the worker checks each remaining proposal independently against previously recorded picks plus clicks actually completed in that operation. It does not use another unselected proposal's planned points to retire a region. Results are applied only to the matching frame and, for exhaustion checks, matching spacing. Zero-pick and final-completion statuses explain what happened instead of always inviting another green-region click. Planning logs record raw, blocked, eligible and sent counts. Sampling, duplicate-distance protection, actual-pick history and the manual callbacks are retained.

A synthetic overlap after five selections reproduces the earlier exact “0 picks placed” status. Eight separated regions already worked in 2.7.1, so this evidence does not establish a universal four- or five-selection limit. The reported micrograph and session logs were not available; the screenshot establishes the displayed message, not its underlying cause. Private regression evidence compares old and new plans across the saved 80-example masks without rerunning SAM or measuring new biological accuracy.

The Windows native fixture exercises eight fresh contours and a covered zero-pick selection introduced after the fifth. It checks actual Tk input events, the worker, recorded pick history, continued selection after zero picks, fresh annotation surfaces, destroyed retired windows, the unchanged topmost panel and final completion status. Source, frozen and installed runtime reports determine whether these native gates passed. They use an in-memory target and send no operating-system input to cryoSPARC. Target-workstation acceptance and the original failure remain outside this synthetic validation.

The 3,500-screen-pixel automatic capsid cutoff, carbon checks, recognition implementation, SAM checkpoint and 80-example appearance profile are unchanged. This is a picking-controller update, not model retraining. Prior 2.7.1 model evidence is identified as reused evidence rather than relabeled as new inference. Unmarked training regions remain unknown; confirmed blank images and explicit unwanted regions would need negative-label support before supervised negative training could be claimed.

# Version 2.7.1 validation scope

This patch adds a configurable final connected-component size gate for both automatic capsid modes. The default is 3,500 foreground screen pixels; holes do not count, disconnected fragments cannot pool area, and an empty result has no fallback to small contours. Manual drawing, explicit prompts, generic detection, proposal generation, SAM weights and the private 80-example appearance weights are retained. Targeted tests cover threshold boundaries, disconnected components, holes, invalid settings, both detectors, empty results, and retiring stale displayed contours after settings changes while preserving actual picks.

Private size-filter evaluation replays the exact production gate on the frozen 2.7.0 held-out masks; it does not represent 80 new SAM runs or newly trained folds. Runtime inference is checked separately. Supplied screenshot outlines are measured as observed detections; a screenshot already containing green contours is not a clean micrograph. Partial-positive examples cannot establish full precision or complete recall. The current release guide contains exact retention and removal measurements.

# Version 2.7.0 validation scope

Version 2.7.0 adds multi-deck training provenance, optional positive interiors
from small marked groups, and optional positive sampling balanced across
screenshots. The public source suite passes **409 tests**. The nine new training
regressions cover incomplete-positive masks, valid-image restrictions, empty
positive cases, sparse-case sampling, unchanged legacy pixel draws, and exact
source-deck assignment and names. Three new prompt regressions cover compact
seeds, preservation of existing compact proposals, and the 24-prompt bound.
Runtime feature extraction, default capsid detection, carbon handling, manual
sampling, screen overlays, picking speed and navigation retain their 2.6.1
behavior. A separate learned-prompt correction handles supported regions that
were previously discarded before SAM could outline a capsid. Application and
installer version identifiers are updated.

The private appearance-profile update uses examples 1–80. Development compares
the shipped 40-example profile on the added examples with held-out 80-example
training candidates, and checks the original examples against their previous
held-out reference. Ten screenshot-group folds hold out eight examples each;
each candidate fold trains on the other 72. The original examples retain their
previous held-out memberships. Feature normalizers use only training samples.
The delivered profile is subsequently fitted on all 80 examples. Neither SAM's
weights nor the inference architecture are changed.

The private reports bind model weights, both source decks, the frozen dataset,
folds, trainer, feature schema, sampling indices and actual positive draws.
Candidate checks use the production SAM prompt/decoding/refinement/filtering
path. Verified image embeddings may be reused, but each candidate computes its
own appearance scores and final proposals. This reuse is not a speed benchmark.
The report distinguishes new-example comparisons, original-example regression
checks, and the final profile's source-runtime integration check.

The supplied markings remain incomplete confident positives. Unmarked cases and
regions are unlabeled, and marker extents are not exact capsid segmentations.
Large-target coverage and contour boundaries have priority in selection, as
requested by the user. Selected area and visual review check overextended
regions; small-target coverage is secondary. These measures do not establish
precision, complete recall or boundary IoU. All candidate selection is internal
development. Original micrograph and
acquisition identifiers were unavailable, so overlap-aware screenshot grouping
cannot establish independence across acquisitions. The private release guide
records measured results, candidate selection and remaining failures.

The Windows release retains the required native, frozen and installed checks
for topmost panel layout, repeated semi-auto selection, micrograph navigation,
shared picking behavior, carbon rejection, model and icon sidecar installation,
and actual SAM inference. Public Windows checks use synthetic nonprivate
fixtures. The delivered private profile is separately checked in the source
runtime. The downloaded Windows reports determine the result of native gates;
source tests alone do not certify the installer or live cryoSPARC behavior.

---

# Version 2.6.1 validation scope

Version 2.6.1 adds a shared, conservative carbon-edge guard after refinement in
both **capsid** and **capsid-learned** automatic detection. It checks whether an
elongated candidate follows a long, coherent edge supported by the surrounding
screen area. An isolated membrane facet, proximity to a line, or membership in a
broad SAM parent is insufficient. Retained masks, holes and proposal order are
unchanged. The guard does not edit the screenshot or exclude an entire side of
an edge. Generic SAM and explicit box/point prompts bypass it.

The SAM checkpoint, learned appearance weights and training examples are
unchanged from 2.6.0. The control layout, manual drawing and sampling, click
pacing, semi-auto selection lifecycle and micrograph navigation behavior are
also unchanged. This release adds geometric rejection, not a newly trained
carbon classifier.

## Paired replay of all 40 development examples

The comparison reuses the exact saved 2.6.0 SAM proposals and held-out appearance
score maps, replays the unchanged refiners and actual final candidate gate, then
adds the new guard between those two stages. Reference replay was verified
against the saved original masks. SAM inference and training were not repeated
for this comparison.

| Measure | Default capsid: before → after | Learned capsid: before → after |
| --- | ---: | ---: |
| Retained contours | 117 → 117 | 292 → 288 |
| Selected valid-image union pixels | 805,278 → 805,278 | 1,305,036 → 1,297,777 |
| Covered supplied positive-core pixels | 312,083 → 312,083 | 416,227 → 416,227 |
| Covered supplied positive marker/ring pixels | 170,650 → 170,650 | 225,724 → 225,724 |
| Positive-core coverage | 61.7973% → 61.7973% | 82.4194% → 82.4194% |
| Positive marker/ring coverage | 53.0086% → 53.0086% | 70.1161% → 70.1161% |

The denominators are 505,011 supplied positive-core pixels and 321,929 positive
marker/ring pixels. Four learned contours were removed in examples 10, 16 and
32, reducing the selected union by 7,259 valid pixels. Independent visual review
within this development session judged them to be edge-following structural
selections without an identifiable compact dense core. No supplied positive
core or marker/ring pixels were lost in either path. The default detector's
outputs did not change on these 40 examples; use of the same guard in both
profiles must not be presented as a measured improvement in both.

**These are incomplete confident positives, not exhaustive target masks or
carbon-negative annotations.** Unmarked detections remain unjudged. The replay
cannot establish precision, complete recall, exact boundary accuracy or
external generalization. Broad detached film selections, compact patches inside
film and curved support remain possible. In particular, a large region can
contain a valid partial capsid: deleting its whole parent would lose known
positive coverage. The guard deliberately retains ambiguous regions.

The bounded context pass averaged 0.1258 seconds per image on this CPU, with a
0.1762-second 95th percentile. Additional contour adjudication averaged 0.0110
seconds for default capsid and 0.0342 seconds for learned capsid: approximately
0.137 and 0.160 seconds total added work, respectively. These measurements omit
SAM inference, screen capture, browser input and navigation and are not Windows
throughput guarantees.

## Source checks and required native Windows gates

The 2.6.1 source suite passes **397 tests**, including 17 carbon-guard regressions
and automatic-routing checks. Coverage includes nearby compact targets,
rotation and resized coordinates, short polygon facets, isolated internal
membrane faces, curved membranes, white margins, noise, retained holes and mask
identity, cancellation, cache behavior, generic/prompt bypass, and a small noisy
field inside otherwise white UI.

The Windows release must additionally pass the native frozen and installed
application gates. Its image-processing smoke check must detect a synthetic
long support edge, reject the known false strip and preserve the neighboring
compact target. Existing checks for the visible topmost panel, successive
semi-auto selections, navigation, icon installation, profile installation,
native libraries and actual SAM inference remain required. This section records
those requirements; the release's downloaded build and installed-runtime
reports determine whether they passed. Synthetic native checks send no real
cryoSPARC input and do not certify browser pick delivery or loading behavior on
the user's workstation.

---

# Version 2.6 validation scope

The 2.6 update separates UI lifecycle validation, optional profile installation, and recognition evaluation. Historical results below apply to their named versions and experiments.

The semi-auto regression requires a fresh native annotation window for each remaining contour set. The old 2.5.0 implementation fails this regression. The native Windows three-selection check also inspects actual Tk image pixels, verifies completed contours are absent, checks two old annotation windows were destroyed, and requires the final annotation to be cleared. The same visible, topmost control panel must survive every selection. These checks establish window lifecycle and current bitmap contents; they are not a recording of every Windows compositor frame or a live cryoSPARC session.

The optional profile has separate strict JSON, feature and inference tests. The frozen public build reports whether the optional profile is available; the default detector retains its texture heuristic. The explicitly selected experimental learned detector requires a valid profile. The installer test supplies a nonprivate constant-weight fixture and verifies that the installed runtime loads its exact bytes and computes the expected numerical scores. No private training images or learned parameters are required in public CI.

Recognition evaluation uses clean images, positive-only markers and unlabeled regions. Learned parameters and normalization must exclude each evaluation group. Marked-location coverage is accompanied by selected area, candidate counts, case-level results and visual inspection. Incomplete annotations cannot establish precision, complete recall or exact boundary overlap. Iterative development comparisons are not an independent external test; the private model's evidence and limitations accompany its delivery.

# Historical SAM validation on the supplied example

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

## Version 2.4.0 navigation and picking speed

This section adds evidence for 2.4.0; the earlier recognition diagnostics and
version-specific test counts above are historical results, not the current
release's test count. Version 2.4.0 does not introduce a new checkpoint, retrain
the recognition model, or claim improved biological recognition accuracy.

### Navigation behavior and release checks

The screen panel replaces **Next alternative** with **Next micrograph** in
semi-auto mode. It sends one Down key to the verified cryoSPARC window, waits
for a different stable image, and detects new contours without picking them.
This navigation/detection path is available while Disable Picking is checked.

Auto mode completes every contour, advances with Down, detects, and repeats.
**Limit (0=all)** defaults to 0 for continuous operation. A positive limit from
1 to 1000 includes the current image and images with no detected contours.
Empty images are counted and skipped. **Load (s)** defaults to 10 seconds and
is a maximum timeout; the app continues when a changed image becomes stable.
An unchanged/end-of-list view, a recently repeated image, a loading timeout,
lost target focus, or an unexpected scene change stops the cycle. Navigation
does not repeatedly send Down while waiting for a response.

The source regression and native Windows release gates exercise navigation
with synthetic image sequences and in-memory input, including continuous and
finite runs, empty images, semi-auto navigation while picking is disabled,
and cancellation/focus failures. The native fixtures use the actual controller,
Tk widgets and worker, but send no operating-system clicks or keys. Source,
frozen and installed test reports from the release workflow determine whether
the build passes these gates. They do not certify real cryoSPARC browser input
delivery or its loading behavior on another workstation.

### Faster model-mode pacing

Semi-auto/auto now default to a **0.03-second delay between clicks**, adjusted
through Advanced in the existing 0.02–2 second range. Manual mode retains its
0.10-second cadence. The playback engine and desktop click adapter are
unchanged: PyAutoGUI's additional global pause remains disabled, its fail-safe
remains enabled, and cancellation, foreground and point-obscuration checks
remain in the per-click path. Targeted regressions exercise focus loss after
the first fast pick, backend fail-safe propagation, and cancellation before the
remaining native calls.

A diagnostic used the actual `play_clicks` and `DesktopAdapter` implementations
with in-memory native/input facades on Linux with Python 3.12.14. Each trial
contained 200 unique picks; two trials were run at each delay. Both rates used
the same explicit simulated costs: 2 milliseconds per click adapter call and
20 milliseconds per scene check, scheduled every 0.75 seconds. No desktop
input was sent.

| Measure | Previous 0.10-second delay | New 0.03-second delay |
| --- | ---: | ---: |
| Median synthetic playback time | 20.885 s | 6.597 s |
| Inter-pick waiting budget, 199 waits | 19.90 s | 5.97 s |
| Point-obscuration checks per trial | 200 | 200 |
| Foreground checks per trial | 401 | 401 |

The synthetic playback measurement was 3.166 times faster, with a 68.41%
reduction in elapsed time. The configured inter-pick waiting budget falls by
70%. Every trial delivered the expected 200 in-memory picks; the minimum
observed spacing at the new setting was 32.16 milliseconds.

**These are synthetic playback measurements, not end-to-end cryoSPARC
throughput.** They exclude actual Windows/browser input delivery, screenshot
capture, model inference, image loading and UI handoff. The simulated costs
are controlled assumptions rather than measurements of those operations.
Browser response and workstation timing determine the practical contour time.

---

## Version 2.5.0 shared picking speed and manual outline cleanup

This section records the 2.5.0 behavior and its validation scope. The earlier
test counts, recognition measurements and 2.4.0 playback benchmark remain
historical results. They do not establish the throughput or test count of
version 2.5.0. This release uses the existing recognition checkpoint and does
not claim improved biological recognition accuracy.

### Shared click pacing

Manual, semi-auto and auto now use the same click-delay setting, defaulting
to **0.01 seconds**. Previously, manual used approximately 0.10 seconds and
the model modes defaulted to 0.03 seconds. A value of **0.00** removes the
configured inter-click wait. Playback still yields execution between clicks
at zero delay, checks for cancellation, and retains the target-focus,
point-obscuration and mouse fail-safe checks.

The setting is available through **Advanced…** in model modes, **F9** while
the SARU panel has focus, or right-clicking the panel and choosing
**Picking speed…**. The compact manual control layout and its original
hold-draw-release gesture are retained. Manual point generation keeps its
original bounding-box origin, integer row spacing, alternating offset,
polygon boundary rule and zero inset.

The configured delay is only one part of elapsed contour time. Browser event
handling, native input delivery, scheduling and guard checks still take time.
Model detection and micrograph loading are separate operations. A setting of
zero is not evidence that the browser registers every submitted click or that
throughput has no limit. Live verification on the intended workstation is
needed to choose the lowest reliable delay.

### Previous-outline flash

Before a new manual trace is displayed, the temporary native annotation
surface is recreated. This removes the previous surface's retained pixels
instead of relying only on deleting its canvas items before showing it again.
The control panel itself is retained and stays visible. This change concerns
the drawing surface lifecycle; it does not change manual sampling geometry.

### Evidence to distinguish

Source regressions and synthetic UI fixtures can check shared delay routing,
zero-delay cancellation, generated-click exclusion, repeated strokes and
the annotation surface lifecycle. Native Windows checks can verify overlay
stacking, hit testing and visibility. Neither type of fixture by itself
measures accepted picks in a real cryoSPARC browser.

Any 2.5.0 timing results must identify the exact input count, delay settings,
hardware or simulated costs, and whether input was sent to a real browser.
Source, frozen and installed reports from the release workflow establish
which gates actually ran and passed; no new test count or measured speedup
is inferred from the changed default alone.



## 3.0.0 offline crYOLO inference

The installation-hang correction exports the existing 160-example H5 checkpoint
to ONNX and bundles ONNX Runtime CPU. It removes automatic legacy runtime setup
from installation and recognition. No model is retrained. The independent worker
validates embedded model metadata, runs the full-frame network, decodes boxes and
uses the unchanged production contour processing. Legacy H5 inference requires
an explicit interpreter override; its optional source setup helper has bounded
lock, download and command waits.

Private verification compares stock TensorFlow outputs with ONNX on all 160
original clean images, including a separate pass through current Pillow/NumPy
preprocessing. Per-case numerical, decoder, contour and picking-point results
belong in the delivered model record. These are conversion-equivalence checks,
not held-out biological validation. Public Windows CI uses a synthetic Conv
network to test positive inference through the frozen and installed worker,
checks exact sidecar copies, and blocks network access during installed checks.
The release record must cite the completed run and its actual reports.
