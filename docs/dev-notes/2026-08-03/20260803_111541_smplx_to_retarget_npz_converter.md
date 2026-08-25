# SMPL-X to Soma Retargeter NPZ Converter

## Scope

This document describes the direct SMPL-X/AMASS/Stage-II to SOMA77 NPZ
conversion path added to Soma Retargeter. The implementation was compared with
NVlabs/SOMA-X commit `86632764684281dc98f31ab9c4aac36a4cdbc428`.

The output is a Kimodo-compatible SOMA77 NPZ that
`app/bvh_to_csv_converter.py` can load and convert to BVH internally before
retargeting.

## Previous Process

The local prototype used four independent Python processes:

1. `smplx_to_soma.py`: SMPL-X parameters to SOMA pose NPZ.
2. `soma_x_npz_to_kimodo_npz.py`: reload the SOMA pose, create another
   `SOMALayer`, and run target forward kinematics.
3. `fix_amass_to_kimodo_frame.py`: reload and convert AMASS coordinates to
   Kimodo coordinates.
4. `rotate_kimodo_heading_180.py`: reload and apply the required heading.

Each stage wrote and reloaded a compressed NPZ. Frame rate was also patched in
a later file instead of being propagated from the source metadata.

## Direct Process

`soma_retargeter.assets.smplx_motion` performs the same logical work in one
process:

1. Normalize combined or split SMPL-X fields.
2. Build the source SMPL-X model and the target SOMA layer once.
3. Run SOMA-X low-LOD `PoseInversion` in batches.
4. Project the recovered rotations to valid SO(3) matrices.
5. Run target SOMA77 forward kinematics without an intermediate NPZ.
6. Apply AMASS-to-Kimodo frame conversion and normalize the frame-zero
   anatomical heading to Kimodo `+Z` in memory.
7. Export one NPZ with motion, identity, frame-rate, coordinate, and fitting
   metadata.

The converter accepts combined `poses` widths 165, 156, and 66, as well as
split AMASS/Stage-II aliases such as `pose_body`, `pose_hand`, `pose_jaw`, and
`pose_eye`. Hand PCA parameters are intentionally unsupported because the
converter requires full axis-angle values.

## Usage

Install the optional SOMA-X dependencies in the active environment:

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com -e '.[soma-x]'
```

Inspect source metadata without initializing the models:

```bash
python tools/convert_smplx_to_retarget_npz.py \
  --input /path/to/source_stageii.npz \
  --output /path/to/output_retarget.npz \
  --model /path/to/SMPLX_NEUTRAL.npz \
  --inspect
```

Convert an AMASS-coordinate motion:

```bash
python tools/convert_smplx_to_retarget_npz.py \
  --input /path/to/source_stageii.npz \
  --output /path/to/output_retarget.npz \
  --model /path/to/SMPLX_NEUTRAL.npz \
  --device cuda:0 \
  --source-coordinate amass
```

Convert every matching NPZ recursively while preserving the input directory
tree:

```bash
python tools/convert_smplx_to_retarget_npz.py \
  --input-dir /path/to/source_motions \
  --output-dir /path/to/retarget_motions \
  --model /path/to/SMPLX_NEUTRAL.npz \
  --device cuda:0
```

Directory mode uses `*.npz` by default. Use `--pattern` to select a narrower
set, for example `--pattern '*_stageii.npz'`. Existing output files are skipped
unless `--force` is supplied. Failed files are reported while the remaining
files continue; `--fail-fast` changes this to stop on the first failure. A run
with any failures exits with status 1 after printing its summary.

Compatible source and target runtimes are cached by model path, gender, beta
count, hand-mean mode, and device. Every motion still prepares its own identity,
but model construction and structural PoseInversion caches are reused. Motions
with a different runtime signature create an additional cache entry instead of
reusing an incompatible model.

Use `--no-compress` when conversion throughput and CPU time are more important
than output size. Existing outputs are never replaced unless `--force` is
given.

The default batch size remains 32. A batch size of 128 reduced the measured
inversion time, but changed recovered poses by as much as `0.136177 rad` in the
validation motion. It was therefore rejected as a behavior-preserving default.

## GUI Frame Rate

`soma_retargeter.assets.kimodo_npz.detect_npz_fps()` now checks the NPZ itself
before the sidecar BVH and the 30 Hz fallback. Explicit environment or config
values still have higher priority.

Priority order:

1. `SOMA_RETARGETER_KIMODO_NPZ_FPS`.
2. `kimodo_npz_fps` in the selected converter config.
3. NPZ metadata (`fps`, `sample_rate`, AMASS frame-rate aliases).
4. Sidecar BVH frame time.
5. Existing 30 Hz fallback.

## Validation

Environment:

- GPU: NVIDIA GeForce RTX 5090.
- Input: 875 frames at 120 Hz.
- Source model: neutral SMPL-X with 16 betas.
- Batch size: 32.
- File writes excluded from both timing paths.

Measured processing time:

| Path | Seconds |
| --- | ---: |
| Previous SMPL-X to SOMA stage | 13.063139 |
| Previous SOMA to Kimodo stage | 5.552000 |
| Previous coordinate conversion | 0.045606 |
| Previous heading conversion | 0.031352 |
| **Previous total** | **18.692097** |
| **Direct total** | **16.667387** |

The direct path reduced measured compute time by `2.024710 s` (`10.83%`). This
comparison excludes file writes, so it does not assign an unmeasured number to
the additional benefit of removing three intermediate compressed NPZ files.

Direct conversion metrics:

- Model initialization: `10.687694 s`.
- Batched conversion: `5.958137 s` (`146.858 frames/s`).
- Coordinate conversion: `0.015249 s`.
- Peak RSS: `2,373,604 KiB`.
- Vertex fit error: mean `0.004561 m`, median `0.002594 m`, maximum
  `0.095214 m`.

Rotation validity:

- Local orthogonality maximum error: `1.1921e-7`.
- Local determinant range: `[0.99999988, 1.00000012]`.
- Global orthogonality maximum error: `1.4901e-6`.
- Global determinant range: `[0.99999857, 1.00000072]`.

Soma Retargeter BVH compatibility:

- Converted Euler shape: `(875, 77, 3)`.
- Matrix-to-Euler-to-matrix maximum error: `3.1885e-13 rad`.
- Mean error: `4.4534e-14 rad`.

Comparison with the previously generated final NPZ:

- Local matrix mean absolute difference: `0.001027`.
- Global matrix mean absolute difference: `0.001390`.
- Posed-joint maximum position difference: `0.002162 m`.
- Posed-joint mean position difference: `0.0000422 m`.
- Root maximum position difference: `0.0000548 m`.
- Frame rate: `120 Hz` in both files.

The previous final file was produced by independent solver processes, so exact
pose equality is not expected. The comparison is reported rather than hidden;
the direct result remains a valid rotation sequence and passes the Retargeter
BVH round-trip check.

## Verification Commands

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover \
  -s tests -p 'test_smplx_motion.py' -v

PYTHONDONTWRITEBYTECODE=1 python -m compileall -q \
  app soma_retargeter tools tests
```

The initial five loader, coordinate, and FPS tests passed. The real 875-frame CUDA
conversion and BVH round-trip validation also completed successfully.

Directory mode added two tests for recursive path preservation and shared
runtime-cache use, bringing the total to seven tests. In a same-process paired
875-frame measurement, the first file took `5.768775 s` and the compatible
second file took `2.285189 s`. Initialization fell from `3.535702 s` to
`0.077781 s` (`97.80%`), and total processing time fell by `60.39%`. These are
warm, paired runtime-reuse measurements and are not substituted for the cold
single-file benchmark above.

## Remaining Constraints

- A licensed SMPL-X model file must be supplied by the user and is not bundled.
- Model initialization dominates single-file execution. The current CLI
  optimizes one-file conversion without changing PoseInversion convergence.
- Directory conversion is sequential. It reuses compatible runtimes but does
  not schedule multiple simultaneous GPU conversions.
- Source expressions are detected but are not transferred to SOMA77.

## 2026-08-03 15:06 KST Update: Input Guardrails From Legacy Assets Pipeline

The legacy scripts under `assets/motions/` were rechecked against the direct
converter:

- `smplx_to_soma.py` converted raw Stage-II/SMPL-X parameters to SOMA-X pose
  data.
- `soma_x_npz_to_kimodo_npz.py` converted SOMA-X pose data into the
  Retargeter/Kimodo NPZ layout.
- `fix_amass_to_kimodo_frame.py` applied AMASS Z-up/+Y-forward to Kimodo
  Y-up/+Z-forward conversion.
- `rotate_kimodo_heading_180.py` applied the final Kimodo +Y heading yaw.

The direct converter still matches that transform structure. A root-channel
check of the legacy final NPZ and the direct NPZ after `NPZ -> BVH` conversion
showed no evidence that another yaw or up-axis correction should be added.

The real risk was that directory mode could pick up intermediate or already
converted NPZ files. That is now guarded:

- Recursive conversion defaults to `*_stageii.npz` instead of `*.npz`.
- `load_smplx_motion()` rejects NPZ files that already contain
  `local_rot_mats`.
- `load_smplx_motion()` rejects SOMA-X intermediate pose files with
  `poses.shape == (T, 77, 3)`.
- GUI NPZ loading no longer forces `kimodo_npz_fps=120`; the default is `null`
  so stored NPZ FPS is used first.

Verification:

```bash
PYTHONPATH=/soma-retargeter /soma-retargeter/.venv/bin/python -m unittest \
  tests.test_smplx_motion -v

PYTHONPATH=/soma-retargeter /soma-retargeter/.venv/bin/python \
  tools/convert_smplx_to_retarget_npz.py \
  --input-dir assets/motions \
  --output-dir /tmp/soma_retargeter_unused \
  --model assets/motions/SMPLX_NEUTRAL.npz \
  --inspect
```

Result:

- 9 unit tests passed.
- Default directory inspection found only `punchboxing_kick_stageii.npz`.
- Already converted Retargeter/Kimodo NPZ files and SOMA-X intermediate NPZ
  files are rejected before PoseInversion starts.

## 2026-08-03 16:27 KST Update: Per-Motion Heading Canonicalization

The earlier fixed `180 deg` heading correction was derived from one
punchboxing sample. It converted that sample to Kimodo `+Z`, but it did not
remove the varying initial `global_orient` stored in other Stage-II files.
Consequently, a converted SOMA mesh could differ from the static K1 model by
approximately 90 degrees immediately after GUI `Load`, before any retargeting.

The converter now determines heading from frame-zero SOMA77 anatomy:

1. Build horizontal left-to-right axes from both shoulder and hip landmarks.
2. Average the valid axes.
3. Cross the lateral axis with Kimodo `+Y` to obtain body forward.
4. Apply the yaw that maps body forward to Kimodo `+Z`.
5. Rotate only root local orientation, all global orientations, and world-space
   positions about the first root position. Child local rotations are retained.

Automatic canonicalization is enabled by default. `--heading-yaw-degrees` is
now an additional offset after canonicalization and defaults to zero.
`--no-canonicalize-heading` is available only for an explicitly unnormalized
export.

Existing converted NPZ files do not require another SOMA-X PoseInversion run.
They can be normalized atomically:

```bash
python tools/normalize_retarget_npz_heading.py \
  --input-dir /path/to/existing_converted_motions \
  --in-place \
  --pattern '*_stageii.npz' \
  --fail-fast
```

Validation used
`BMLrub/rub001/0005_normal_walk1_stageii.npz` (252 frames, 120 Hz):

- Before normalization, GUI source/K1 heading difference: `88.130605 deg`.
- Applied Kimodo yaw: `-88.130605 deg` for the existing converted file.
- After normalization, GUI source/K1 heading difference: `0.0 deg`.
- Child local-rotation maximum change: `0.0`.
- Pairwise joint-distance maximum change: `2.3842e-7 m`.
- A direct conversion from the original Stage-II file also produced body
  forward `[8.61e-8, 0, 1]`, effectively Kimodo `+Z`.

All 2,516 existing `*_stageii.npz` outputs contained valid shoulder/hip
landmarks for automatic heading calculation. Of those files, 969 required
more than 5 degrees of additional correction. The files were scanned but were
not overwritten during validation.

The default AI Sapiens target orientation yaw remains `0.0`; source heading is
corrected at conversion/load-data level rather than hidden in an IK target
orientation setting.

## 2026-08-03 17:10 KST Update: Frame-Zero Horizontal Origin

The GUI does not automatically align a loaded SOMA source root with the K1
MJCF root. Existing converted motions retained a non-zero frame-zero root X/Z,
so source and target could start at different horizontal positions even before
retargeting.

Horizontal rebasing is now part of the same converter layer that performs
heading canonicalization. It subtracts
`[root_positions[0, X], 0, root_positions[0, Z]]` from every world-space
position field:

- `posed_joints`
- `root_positions`
- `smooth_root_pos`
- `world_root_position`
- `root_translation`
- `transl`
- `trans`

Kimodo Y height, frame count, FPS, relative root trajectory, rotations, and
joint geometry are retained. New conversions enable the behavior by default;
`--no-rebase-root-horizontal` exists for explicitly preserving source global
placement.

This logic is implemented in Soma Retargeter, not by modifying the installed
NVlabs SOMA-X package:

- `soma_retargeter/assets/smplx_motion.py` contains the common heading and
  position normalization.
- `tools/convert_smplx_to_retarget_npz.py` uses it during new SOMA-X
  PoseInversion conversions.
- `tools/normalize_retarget_npz_heading.py` applies the same normalization to
  existing converted NPZ files without rerunning SOMA-X or PoseInversion.

Normalize all existing converted Stage-II outputs in place:

```bash
cd /soma-retargeter
PYTHONPATH=/soma-retargeter /.venv/bin/python \
  tools/normalize_retarget_npz_heading.py \
  --input-dir /motion-output \
  --in-place \
  --pattern '*_stageii.npz' \
  --fail-fast
```

Validation on `BMLrub/rub001/0005_normal_walk1_stageii.npz` produced:

- frame-zero root before: `[0.033863384, 1.082554936, -0.003066148] m`
- frame-zero root after: `[0.0, 1.082554936, 0.0] m`
- output anatomical heading: `[approximately 0, 0, 1]`
- height maximum change: `0.0 m`
- relative root trajectory maximum change: `2.3283e-10 m`
- child local-rotation maximum change: `0.0`
- pairwise joint-distance maximum change: `2.3842e-7 m`
- second-pass position maximum change: `7.4506e-9 m`
- frame count/FPS retained: `252 / 120 Hz`

Validation artifacts are stored outside the repository at:

```text
/media/hc/82C4B899C4B89141/soma_x/20260803_170846_heading_position_normalization_validation/
```

## 2026-08-03 17:39 KST Operational Check: Container Version and FPS Override

`back_filp.npz` exposed two runtime-version problems that are independent of
the source motion metadata:

- The source has 120 frames at 30 Hz (`4.0 s`). The converted output also
  stores 120 frames at 30 Hz (`4.0 s`), so conversion preserves its FPS.
- The `crazy_spence` container config still sets `kimodo_npz_fps` to `120.0`.
  GUI Load therefore rebuilds the temporary BVH at 120 Hz and displays a
  `1.0 s` duration, four times faster than the source.
- The same container runs the earlier converter with a fixed
  `heading_yaw_degrees=180.0`. Its output anatomical forward is approximately
  Kimodo `-Z`, opposite the required `+Z`.
- The current dynamic heading and horizontal-origin implementation was present
  in `crazy_spence_mounted`, where `kimodo_npz_fps` is `null`, but had not been
  synchronized to `crazy_spence`.

Before converting or viewing a motion, the converter implementation and
default config must come from the same updated working tree. An output NPZ's
FPS fields alone cannot override an explicit non-null `kimodo_npz_fps` viewer
configuration.

## 2026-08-04 08:04 KST Update: Kimodo SMPL-X 22-Joint Matrix Input

Kimodo SMPL-X downloads can use an evaluated body-matrix schema rather than
the AMASS/Stage-II parameter schema. The observed schema contains:

```text
local_rot_mats   (T, 22, 3, 3)
global_rot_mats  (T, 22, 3, 3)
posed_joints     (T, 22, 3)
root_positions   (T, 3)
foot_contacts    (T, 4)
```

This is neither an already converted SOMA77 motion nor a K1 retargeting
result. The previous loader incorrectly rejected every file containing
`local_rot_mats` without checking the joint dimension.

The converter now distinguishes formats by schema:

- 22 matrices: Kimodo SMPL-X root plus 21 body joints; convert to SOMA77.
- 77 matrices: already converted Soma Retargeter input; load in the GUI.
- AMASS/Stage-II pose parameters: retain the existing conversion path.

For 22-joint input, rotation matrices are projected to valid SO(3) axis-angle
parameters. Root positions are converted to SMPL-X translations using the
pelvis offset computed from the selected body model, without a motion-specific
offset. `--source-coordinate auto` selects Kimodo coordinates for this schema
and AMASS coordinates for Stage-II parameters. Existing anatomical heading
canonicalization and horizontal root rebasing remain responsible for the
final Kimodo `+Z` heading and frame-zero X/Z origin.

Kimodo matrix exports may omit frame-rate metadata. The compatibility default
remains 30 Hz, and `--input-fps` provides an explicit override:

```bash
PYTHONPATH=/soma-retargeter /.venv/bin/python \
  tools/convert_smplx_to_retarget_npz.py \
  --input /path/to/kimodo_smplx_22_joint.npz \
  --output /path/to/soma77_output.npz \
  --model /soma-retargeter/assets/motions/SMPLX_NEUTRAL.npz \
  --device cuda:0 \
  --input-fps 30
```

Validation covered two real matrix exports and one Stage-II regression:

- `rolling`: 121 input/output frames, 77 output joints, Kimodo `+Z` heading.
- `side_flip`: 89 input/output frames, 77 output joints, Kimodo `+Z` heading.
- Stage-II smoke: AMASS auto-detection and 60 Hz metadata preserved.
- 17 unit tests and Python compilation passed.

The 22-joint schema does not include finger pose or identity betas. Conversion
therefore uses neutral hand pose and the supplied neutral SMPL-X model's zero
betas. Validation artifacts and metrics are stored at:

```text
/media/hc/82C4B899C4B89141/soma_x/20260804_080009_kimodo_smplx22_conversion_validation/
```
