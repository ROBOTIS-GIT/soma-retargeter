# SOMA-X Preprocessing Integration

Created: 2026-08-24 12:11:49 KST (Asia/Seoul)

## Scope

SOMA-X is integrated as an optional preprocessing stage for Soma Retargeter:

```text
raw SMPL/SMPL-H/SMPL-X NPZ -> SOMA-X PoseInversion -> SOMA77 NPZ
                                                     -> optional BVH
SOMA77 NPZ/BVH -> existing Soma Retargeter pipeline -> robot CSV
```

The GUI and standalone preprocessing entrypoints remain available. For
headless operation, `app/bvh_to_csv_converter.py` also owns the complete
preprocessing and K1 CSV pipeline. It classifies each input by file schema and
invokes SOMA-X only for raw SMPL-family motion. Existing BVH config-only batch
operation and the GUI BVH/NPZ rows remain unchanged.

## Optional Installation

The base package does not import SOMA-X, Torch, or SMPL-X unless conversion is
requested. Install the optional runtime from the repository root with one of:

```bash
python -m pip install --extra-index-url https://pypi.nvidia.com -e '.[soma-x]'
```

```bash
uv sync --extra soma-x
```

The extra pins `py-soma-x[smpl]==0.2.1`. A licensed SMPL-family model file is
also required. The model is not included in this repository and must be
obtained and used under its own license. Supported source model families are
`SMPL`, `SMPL-H`, and `SMPL-X`; `.npz` and `.pkl` model files are accepted.
Male, female, and neutral model files are supported, but the selected file is
authoritative and is never replaced silently from motion gender metadata.

Model path precedence is:

1. Unified CLI `--human-model`, standalone CLI `--model`, or the model selected
   in the current GUI session
2. `SOMA_RETARGETER_SOMA_X_HUMAN_MODEL`
3. `soma_x_human_model` in the application config

For existing SMPL-X setups, `SOMA_RETARGETER_SOMA_X_SMPLX_MODEL` and
`soma_x_smplx_model` remain fallback settings.

Model type is detected from topology and parameter structure (`v_template`,
`kintree_table`, hand fields, and face fields), not from the filename. The
matching `smplx.SMPL`, `smplx.SMPLH`, or `smplx.SMPLX` class is instantiated
before a GUI selection is accepted.

## GUI Workflow

Run the existing application:

```bash
python ./app/bvh_to_csv_converter.py \
  --config ./assets/default_ai_sapiens_bvh_to_csv_converter_config.json \
  --viewer gl
```

The Motion panel has one independent row:

```text
SOMA-X: [Human Model] [Motion] [Retarget]
```

1. At GUI startup, Soma Retargeter checks the optional SOMA-X runtime. When the
   required runtime is available, only `Human Model` is initially enabled and
   the line directly below the button row displays
   `Supported models: SMPL / SMPL-H / SMPL-X`. When SOMA-X is missing, has the
   wrong version, or lacks a required module, all three buttons are disabled.
   The same information line displays the reason and the `.[soma-x]` editable
   install command in red. Restart the GUI after installing the runtime.
2. Select a licensed SMPL, SMPL-H, or SMPL-X model. The information line then
   displays the detected family and filename in a smaller normal-foreground
   font, for example `SMPL-X | SMPLX_NEUTRAL.npz`. Hover over the line to see
   the full model path and structural details. A valid selection also enables
   `Motion`.
3. Select a raw motion with `Motion`. Selection immediately starts conversion
   in a subprocess. All three buttons are locked while it runs. Progress is
   printed to the terminal. The model information line appends only the frame
   conversion percentage, such as `| 37%`; no textual runtime status is shown.
4. Successful conversion loads the SOMA77 result through the existing NPZ to
   BVH path, leaves `100%` visible, and enables `Retarget`. A failed or
   cancelled conversion clears the percentage.
5. `Retarget` runs only the existing robot retarget operation. Selecting a new
   human model invalidates the previous motion and converted output.

Before conversion starts, the selected model family is checked against the
motion parameter schema. A recognized mismatch does not open a traceback
popup; the information line instead displays
`Human model and motion do not match.` in red. The selected model remains,
`Motion` is re-enabled, and `Retarget` remains disabled. Other conversion
failures continue to use the existing popup path.

The GUI conversion process is terminated when the viewer closes. The existing
`BVH Motion` and `NPZ Motion` rows retain their original behavior.

## Unified Headless CLI

`bvh_to_csv_converter.py` accepts one motion file or recursively scans one
directory. No `--input-format` option is required: the implementation inspects
the extension and NPZ fields before performing any conversion.

Retarget a BVH or an already converted SOMA77/Kimodo NPZ:

```bash
python app/bvh_to_csv_converter.py \
  --config assets/default_ai_sapiens_bvh_to_csv_converter_config.json \
  --viewer null \
  --input /data/motion.bvh \
  --output-dir /data/k1-output
```

Retarget raw SMPL, SMPL-H, or SMPL-X motion through SOMA-X in the same command:

```bash
python app/bvh_to_csv_converter.py \
  --config assets/default_ai_sapiens_bvh_to_csv_converter_config.json \
  --viewer null \
  --input /data/motion_stageii.npz \
  --human-model /models/SMPLX_NEUTRAL.npz \
  --output-dir /data/k1-output
```

Use the same command with a directory to process all `.bvh` and `.npz` files
recursively while preserving their relative paths:

```bash
python app/bvh_to_csv_converter.py \
  --config assets/default_ai_sapiens_bvh_to_csv_converter_config.json \
  --viewer null \
  --input /data/motions \
  --human-model /models/SMPLX_NEUTRAL.npz \
  --output-dir /data/k1-output \
  --device cuda:0
```

The output layout is deterministic:

```text
/data/k1-output/
  soma_npz/       # retained SOMA-X output for raw SMPL-family input
  soma_bvh/       # retained SOMA BVH for every NPZ input
  retargeted_csv/ # final K1 CSV for every accepted input
```

The input directory and output directory cannot overlap. One directory run uses
one selected human model. All raw inputs are preflighted against that detected
model family before the first SOMA-X conversion starts; a mixed-family mismatch
fails the run instead of partially converting it. A human model file inside the
input tree is excluded only when it is the exact file selected by
`--human-model`. Other unsupported NPZ schemas fail explicitly. Inputs with the
same relative stem, such as `walk.bvh` and `walk.npz`, are rejected because they
would map to the same final CSV.

Relevant optional overrides are:

- `--input-fps`: overrides raw-motion FPS and NPZ-to-BVH FPS.
- `--soma-x-batch-size`: overrides the SOMA-X frame batch size.
- `--source-coordinate {auto,amass,kimodo}`: overrides raw source coordinates.
- `--canonicalize-heading` / `--no-canonicalize-heading`.
- `--heading-yaw-degrees`.
- `--rebase-root-horizontal` / `--no-rebase-root-horizontal`.
- `--bvh-template`, `--bvh-offsets`, and `--bvh-position-scale`.
- `--force`: replaces a stale or invalid retained SOMA77 intermediate.

When `--device` is supplied, the same device is used for SOMA-X and Newton.
When omitted, SOMA-X retains `soma_x_device` from the config, whose default is
`auto`. SOMA-X is imported only when at least one raw SMPL-family input exists;
BVH and SOMA77 input do not require the optional SOMA-X dependencies or a human
model.

If neither `--input` nor `--output-dir` is supplied, the original config-only
BVH batch behavior remains active:

```bash
python app/bvh_to_csv_converter.py \
  --config /path/to/batch_config.json \
  --viewer null \
  --device cuda:0
```

## Standalone Preprocessing CLI

Use `tools/convert_smpl_to_retarget_npz.py` when only the SOMA77 intermediate
is needed. It calls the same package-level conversion function used by the
unified headless entrypoint:

```bash
python tools/convert_smpl_to_retarget_npz.py \
  --input /data/motion_stageii.npz \
  --output /data/converted/motion_stageii.npz \
  --model /models/SMPLX_NEUTRAL.npz \
  --model-type auto \
  --device auto
```

Its existing `--input-dir`, `--emit-bvh`, `--exclude-dir`, and `--inspect`
options remain available. `tools/convert_smplx_to_retarget_npz.py` remains a
compatibility entrypoint that defaults to `--model-type smplx` and delegates to
the same implementation.

## Input And Output Contract

Supported raw inputs are:

- SMPL combined 72-value axis-angle pose or equivalent separate fields
- SMPL-H combined 156-value pose; 66-value body-only pose uses neutral hands
- SMPL-X combined 165-, 156-, or 66-value pose and equivalent separate fields
- Kimodo SMPL-X matrix exports with 22 `local_rot_mats` joints

SMPL uses a 69-value body pose. SMPL-H and SMPL-X use a 63-value body pose;
full hand fields are 45 values per hand. SMPL-X face and expression fields are
preserved when supplied. Hand/face fields that do not belong to the selected
model family fail explicitly instead of being ignored.

The standalone preprocessor rejects already converted 77-joint SOMA77 NPZ
files with instructions to load them directly. The unified headless entrypoint
accepts those files and proceeds directly through fixed BVH generation and K1
retargeting. A `(T, 77, 3)` SOMA-X intermediate pose file is rejected because
it is neither a raw SMPL-family source nor a final Soma Retargeter matrix
schema.

SMPL and SMPL-H use SOMA-X's SMPL topology transfer backend; SMPL-X uses the
SMPL-X backend. SMPL-H is validated and posed with `smplx.SMPLH`, while its
6890-vertex identity topology is transferred through the SMPL backend because
`py-soma-x==0.2.1` does not ship a separate `SMPLH/` transfer-asset directory.

Output contains local/global `(T, 77, 3, 3)` rotation matrices, SOMA77 joint
positions, root data, source metadata, FPS aliases, and a conversion signature.
Generic metadata includes `source_model_type`, `human_model_file`, and
`source_num_betas`. SMPL-X output also retains `smplx_model_file` for backward
compatibility.
Writes use PID-specific temporary files followed by an atomic replace.

## FPS And Coordinate Rules

FPS precedence is:

1. `--input-fps`
2. scalar internal metadata: `mocap_frame_rate`, `mocap_framerate`, `fps`,
   `sample_rate`, `frame_rate`, `framerate`, or `source_fps`
3. compatibility fallback of 30 FPS when no valid metadata exists

The selected FPS is copied to all supported output aliases and is reused by the
GUI NPZ loader and optional BVH writer.

With `--source-coordinate auto`, raw parameter NPZ files use the AMASS frame
conversion and 22-joint Kimodo matrix files retain the Kimodo frame. By default:

- frame-zero anatomical forward is rotated to Kimodo `+Z`
- frame-zero root `X/Z` is moved to the horizontal origin
- vertical position and all relative trajectory displacement are preserved

Use `--no-canonicalize-heading` or `--no-rebase-root-horizontal` only when the
source already has a deliberately different world-frame contract.

## Resume And Cache Rules

Each raw-conversion output stores a SHA-256 signature derived from source and
model path/size/mtime, selected model type, pinned SOMA-X version, and
conversion options. The unified and standalone paths both use the same
signature implementation. An existing SOMA77 intermediate is reused only when
its required fields and signature match. Stale or invalid output fails
explicitly; `--force` is required to replace it. Retained BVH is regenerated
from the accepted SOMA77 NPZ, so PoseInversion is not repeated.

## Troubleshooting

- Missing optional runtime: install `.[soma-x]` and keep version `0.2.1`.
- Missing model: pass `--human-model` to the unified command or `--model` to the
  standalone preprocessor, set `SOMA_RETARGETER_SOMA_X_HUMAN_MODEL`, or
  configure `soma_x_human_model`.
- Model/motion mismatch: select the matching family or provide a motion schema
  compatible with the selected model. The GUI reports this mismatch inline in
  red without launching conversion; the converter does not silently coerce one
  family into another.
- Compact SMPL-H archives can omit MANO PCA basis and mean arrays. The
  converter supplies neutral runtime-only values because hand input is passed
  as full 45-value axis-angle poses with PCA disabled. The `smplx` warning
  about 10 SMPL+H shape coefficients is informational; conversion uses the
  same effective 10-beta dimension selected by that runtime.
- Requested CUDA unavailable: use `--device cpu` or expose the GPU to the
  container.
- BVH offset error: configure `kimodo_npz_offsets` or pass `--bvh-offsets` with
  `standard_t_pose_global_offsets_rots.p`. NPZ-only conversion does not need
  this BVH rest-pose file.
- Existing output signature mismatch: inspect the changed input/model/options,
  then rerun intentionally with `--force`.

## Model Information Display

The line below the `SOMA-X` controls uses an opaque white foreground at a
fixed integer 12-pixel font size. Using an integer size avoids the blurred
glyph edges produced by scaling ImGui's 13-pixel default font to 80 percent.

Before model selection, the line shows:

```text
Supported models: SMPL / SMPL-H / SMPL-X
```

After selection, only the model family, file name, and conversion percentage
are kept on the line:

```text
SMPL-X | SMPLX_NEUTRAL.npz | 37%
```

Hovering the line shows the full model path, vertex count, joint count, and
shape-coefficient count without crowding the fixed-width Scene Options panel.
When the selected model and motion schemas do not match, the same line is
replaced by the red message:

```text
Human model and motion do not match.
```

## Verification Record

Validation on 2026-08-24 covers the common model implementation as follows:

- 60 unit/GUI/headless tests passed, including structural detection for all three model
  families, schema mismatch failures, compatibility helpers, button-state
  transitions, failure recovery, removal of runtime status text, and the
  dedicated 12-pixel high-contrast model information line, shortened selected
  model text, detailed hover tooltip, pre-selection supported model text,
  frame conversion percent, red inline model/motion mismatch reporting, and
  semantic `soma_npz/`, `soma_bvh/`, `retargeted_csv/` output paths.
- Actual licensed SMPL `.npz`, compact SMPL-H `.npz`, and SMPL-X `.npz`
  models were structurally detected and instantiated through their matching
  source classes.
- Actual CUDA conversion produced one SOMA77 frame for SMPL and SMPL-X at the
  source 30 FPS. Both outputs had `(1, 77, 3, 3)` local rotations and generic
  model metadata.
- The generic SMPL-X output was converted to BVH and consumed by the existing
  headless K1 pipeline, which wrote `/tmp/soma_x_k1_smoke/csv/smplx_smoke.csv`.
- The legacy SMPL-X CLI completed a separate actual CUDA conversion.
- An actual compact SMPL-H neutral model from
  `/home/hc/Downloads/smplh.tar.xz` completed a one-frame CPU PoseInversion
  conversion. The output had `(1, 77, 3, 3)` local rotations, 30 FPS, and 10
  effective identity coefficients, matching the installed `smplx` runtime.
- Unified headless smoke tests completed for direct BVH, existing SOMA77 NPZ,
  raw SMPL-H NPZ on CPU, and raw SMPL-H NPZ with automatic CUDA selection.
- Direct BVH output through the unified CLI and the original config-only batch
  path had the same CSV SHA-256
  `147ef277c8b8076eba9d66b1d17445e22dc8aa6e4ae4686016c022f987c219c8`.
- Raw SMPL-H conversion through the common in-process API matched the previous
  standalone SOMA77 reference with `max_abs_diff=0.0` for local/global
  rotations, posed joints, root position, and FPS. Their generated BVH and K1
  CSV files were byte-identical.
- Repeating the raw CPU command reported `matching conversion signature`,
  reused the retained SOMA77 NPZ, and emitted no SOMA-X frame-conversion step.

The SMPL-H verification artifacts are stored under
`/media/hc/82C4B899C4B89141/soma_x/20260824_154143_smplh_load_validation/`.
Unified headless verification artifacts and hashes are stored under
`/media/hc/82C4B899C4B89141/soma_x/20260824_173755_unified_headless_cli/`.
