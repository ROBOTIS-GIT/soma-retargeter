# SOMA-X Preprocessing Integration

Created: 2026-08-24 12:11:49 KST (Asia/Seoul)

## Scope

SOMA-X is integrated as an optional preprocessing stage for Soma Retargeter:

```text
raw SMPL/SMPL-H/SMPL-X NPZ -> SOMA-X PoseInversion -> SOMA77 NPZ
                                                     -> optional BVH
SOMA77 NPZ/BVH -> existing Soma Retargeter pipeline -> robot CSV
```

The preprocessing and robot retargeting commands remain separate. Raw SMPL-X
input is not passed directly to the K1 CSV exporter. Existing BVH and Kimodo
SOMA77 NPZ paths are unchanged.

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

1. CLI `--model` or the model selected in the current GUI session
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

1. Initially only `Human Model` is enabled and the information text is blank.
2. Select a licensed SMPL, SMPL-H, or SMPL-X model. A valid selection displays
   only `SMPL`, `SMPL-H`, or `SMPL-X` and enables `Motion`.
3. Select a raw motion with `Motion`. Selection immediately starts conversion
   in a subprocess. All three buttons are locked while it runs. Progress is
   printed to the terminal; no runtime status text is shown in the GUI.
4. Successful conversion loads the SOMA77 result through the existing NPZ to
   BVH path and enables `Retarget`.
5. `Retarget` runs only the existing robot retarget operation. Selecting a new
   human model invalidates the previous motion and converted output.

On conversion failure, the selected model remains, `Motion` is re-enabled,
and `Retarget` remains disabled. Errors use the existing popup path.

The GUI conversion process is terminated when the viewer closes. The existing
`BVH Motion` and `NPZ Motion` rows retain their original behavior.

## CLI Workflow

The canonical CLI is `convert_smpl_to_retarget_npz.py`. Inspect a motion and a
selected model:

```bash
python tools/convert_smpl_to_retarget_npz.py \
  --input /data/motion_stageii.npz \
  --model /models/SMPLX_NEUTRAL.npz \
  --model-type auto \
  --inspect
```

Convert one motion to SOMA77 NPZ:

```bash
python tools/convert_smpl_to_retarget_npz.py \
  --input /data/motion_stageii.npz \
  --output /data/converted/motion_stageii.npz \
  --model /models/SMPLX_NEUTRAL.npz \
  --model-type auto \
  --device auto
```

Convert one motion and also emit BVH:

```bash
python tools/convert_smpl_to_retarget_npz.py \
  --input /data/motion_stageii.npz \
  --output /data/converted/motion_stageii.npz \
  --model /models/SMPLX_NEUTRAL.npz \
  --model-type auto \
  --device auto \
  --emit-bvh \
  --bvh-output /data/converted-bvh/motion_stageii.bvh
```

Convert a directory recursively while preserving relative paths:

```bash
python tools/convert_smpl_to_retarget_npz.py \
  --input-dir /data/amass \
  --output-dir /data/soma77 \
  --pattern '*_stageii.npz' \
  --model /models/SMPLX_NEUTRAL.npz \
  --model-type auto \
  --device cuda:0 \
  --emit-bvh \
  --bvh-output-dir /data/soma77-bvh
```

`--model-type` accepts `auto`, `smpl`, `smplh`, or `smplx`. One directory run
uses one selected model; mixed-model batches are rejected by schema checks.
The recursive converter reuses compatible SOMA-X model/runtime objects between
motions. `--exclude-dir RELATIVE_PATH` may be repeated to omit input subtrees.

`tools/convert_smplx_to_retarget_npz.py` remains as a compatibility entrypoint.
It defaults to `--model-type smplx` and uses the same conversion implementation.

To continue into the existing headless robot retargeter, configure the emitted
BVH root as `import_folder`, configure the CSV destination as `export_folder`,
then run:

```bash
python app/bvh_to_csv_converter.py \
  --config /path/to/batch_config.json \
  --viewer null \
  --device cuda:0
```

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

Already converted 77-joint SOMA77 NPZ files are rejected with instructions to
load them directly. A `(T, 77, 3)` SOMA-X intermediate pose file is also
rejected because it is not a raw SMPL-X source and does not contain the final
Soma Retargeter matrix schema.

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

Each output stores a SHA-256 conversion signature derived from source and model
path/size/mtime, selected model type, pinned SOMA-X version, and conversion options. An existing
output is reused only when required SOMA77 fields and the signature match.
Stale or invalid outputs fail explicitly; `--force` is required to replace
them. If the NPZ is valid but an optional BVH is missing, BVH can be emitted
without running PoseInversion again.

## Troubleshooting

- Missing optional runtime: install `.[soma-x]` and keep version `0.2.1`.
- Missing model: pass `--model`, set `SOMA_RETARGETER_SOMA_X_HUMAN_MODEL`, or
  configure `soma_x_human_model`.
- Model/motion mismatch: select the matching family or provide a motion schema
  compatible with the selected model; the converter does not silently coerce
  one family into another.
- Requested CUDA unavailable: use `--device cpu` or expose the GPU to the
  container.
- BVH offset error: configure `kimodo_npz_offsets` or pass `--bvh-offsets` with
  `standard_t_pose_global_offsets_rots.p`. NPZ-only conversion does not need
  this BVH rest-pose file.
- Existing output signature mismatch: inspect the changed input/model/options,
  then rerun intentionally with `--force`.

## Verification Record

Validation on 2026-08-24 covers the common model implementation as follows:

- 40 unit/GUI tests passed, including structural detection for all three model
  families, schema mismatch failures, compatibility helpers, button-state
  transitions, failure recovery, and removal of runtime status text.
- Actual licensed SMPL `.npz` and SMPL-X `.npz` models were structurally
  detected and instantiated through their matching source classes.
- Actual CUDA conversion produced one SOMA77 frame for SMPL and SMPL-X at the
  source 30 FPS. Both outputs had `(1, 77, 3, 3)` local rotations and generic
  model metadata.
- The generic SMPL-X output was converted to BVH and consumed by the existing
  headless K1 pipeline, which wrote `/tmp/soma_x_k1_smoke/csv/smplx_smoke.csv`.
- The legacy SMPL-X CLI completed a separate actual CUDA conversion.
- No licensed SMPL-H model file exists in the current container. SMPL-H model
  detection, motion parsing, backend dispatch code, and mismatch handling are
  unit-tested, but a real SMPL-H PoseInversion run remains unverified until a
  licensed model is supplied.

Smoke files are under `/tmp` and are not repository deliverables.
