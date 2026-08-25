# SOMA77 Rest-Pose Offsets Packaging

## Record

- Date: 2026-08-25 10:58:59 KST
- Timezone: Asia/Seoul
- Repository: `/soma-retargeter`
- Branch: `feature/soma-x-integration`

## Problem

The GUI could load an SMPL-H identity model and start converting a raw motion
with SOMA-X, but conversion stopped before loading the result into the viewer:

```text
Kimodo SOMA77 rest-pose offsets file not found.
```

The failure occurred in the SOMA77 NPZ-to-fixed-BVH stage, not in SMPL-H model
validation or SOMA-X topology conversion.

## Root Cause

`soma_retargeter.assets.kimodo_npz.find_default_offsets_path()` already looked
for a package-local standard rest-pose offsets file, but that file was absent
from the repository and from setuptools package data. The running development
container happened to have a copy under `/kimodo`, while a clean
Soma Retargeter installation did not.

Hard-coding `/kimodo` in the application config would only repair that one
container. Reconstructing the matrices from BVH `OFFSET` fields was also not
equivalent to the reference asset; the naive reconstruction showed rotations
differing by up to approximately 180 degrees.

## Resolution

The exact Kimodo reference asset is now packaged at:

```text
soma_retargeter/configs/soma/standard_t_pose_global_offsets_rots.p
```

`pyproject.toml` includes `configs/**/*.p`, so editable installs and built
wheels both provide the fallback. The Kimodo Apache-2.0 license is retained as
`licenses/kimodo-LICENSE.txt`.

Source provenance:

- Kimodo repository source snapshot used for the packaged copy:
  `6bb58488037dd65360ff0c5d1692b403a23309f7`
- Asset first introduced in Kimodo commit:
  `032b4fc2ec32716f512b8293d31eccc4e67ab52d`
- Asset SHA-256:
  `464a8a95159d5e26ad24a702107aec86698935deeb034e02c9fe51e55472d75b`
- Modification status: byte-for-byte copy; no modification

The wheel also contains `licenses/kimodo-ATTRIBUTION.txt`, which records the
source URL, source path, commits, hash, modification status, and license path.

No absolute container path was added to either default converter config.

## Verification

The regression suite verifies that:

1. Default lookup resolves the package-local asset.
2. The asset contains 77 valid 3x3 rotation matrices.
3. A SOMA77 NPZ converts to BVH without `kimodo_npz_offsets` configuration.
4. Existing SOMA-X GUI, model parsing, and headless input tests still pass.
5. A built wheel contains the offsets asset, Kimodo license, and Kimodo
   attribution record.

Commands:

```bash
PYTHONPATH=/soma-retargeter /.venv/bin/python -m unittest \
  tests.test_kimodo_npz \
  tests.test_soma_x_gui \
  tests.test_smplx_motion \
  tests.test_headless_motion_input

/.venv/bin/python -m pip wheel . \
  --no-deps \
  --no-build-isolation \
  --wheel-dir /tmp/<wheel-directory>
```

Result: 64 tests passed, and wheel-content inspection found both required
files.

## Remaining Verification

The reported `Trial_upper_left_005_poses.npz` was not present inside the
container during diagnosis, so that exact motion was not rerun. Retrying it
after restarting the GUI is still required to verify the full user workflow.
