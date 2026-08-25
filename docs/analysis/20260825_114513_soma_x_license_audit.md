# Soma Retargeter SOMA-X License Audit

## Record

- Date: 2026-08-25 11:45:13 KST
- Timezone: Asia/Seoul
- Repository: `/soma-retargeter`
- Branch: `feature/soma-x-integration`
- Base: `origin/main` at `8dc838e`
- Audited branch head before this report: `133cd90`

This is an engineering distribution audit, not a legal opinion. It records the
files, dependency metadata, upstream license texts, and package contents that
were directly checked.

## Scope

The audit covers the SOMA-X integration added after `origin/main`, the current
uncommitted packaging fix, the optional Python dependency graph, the bundled
Kimodo rest-pose asset, and accidental inclusion of user-provided identity
models or motion files.

Official references:

- [SOMA-X license](https://github.com/NVlabs/SOMA-X/blob/main/LICENSE)
- [SOMA-X repository and model-file warning](https://github.com/NVlabs/SOMA-X)
- [SOMA-X runtime asset card](https://huggingface.co/nvidia/soma-x)
- [Kimodo license](https://github.com/nv-tlabs/kimodo/blob/main/LICENSE)
- [SMPL-X software license](https://github.com/vchoutas/smplx/blob/main/LICENSE)
- [SMPL-X model download terms](https://smpl-x.is.tue.mpg.de/register.php)

## Findings

| Component | Distribution in this repository | License finding | Result |
| --- | --- | --- | --- |
| Soma Retargeter source | Source and wheel | Apache-2.0 | Compatible with the repository license. |
| `py-soma-x==0.2.1` | Optional dependency reference; not copied into the wheel | Package metadata and upstream `LICENSE` identify Apache-2.0. | No Apache license conflict found. |
| SOMA-X runtime assets | Downloaded at runtime into the Hugging Face cache; not committed or packaged here | Official `nvidia/soma-x` model card identifies Apache-2.0. | Not redistributed by this repository. |
| Kimodo SOMA77 offsets | One unmodified `.p` asset is packaged | Kimodo is Apache-2.0. The exact upstream license and a source attribution record are shipped in the wheel. | Redistributable under the recorded Apache terms. |
| `smplx==0.1.28` software | Installed only through the optional `soma-x` extra; not copied into this wheel | Max Planck non-commercial scientific research license. | Optional SMPL workflows are not generally commercially permissive without an appropriate separate license. |
| SMPL/SMPL-H/SMPL-X identity models | User-supplied paths only | Separate model terms apply and redistribution is restricted. | No identity model file is committed or packaged. |
| User motion/test data | Untracked workspace files only | Provenance varies by user dataset. | Explicitly excluded from staging and package contents. |

## Kimodo Asset Provenance

The packaged file is:

```text
soma_retargeter/configs/soma/standard_t_pose_global_offsets_rots.p
```

It was compared byte-for-byte with:

```text
/kimodo/kimodo/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p
```

The source checkout was at Kimodo commit
`6bb58488037dd65360ff0c5d1692b403a23309f7`. Git history shows that the asset
was introduced at `032b4fc2ec32716f512b8293d31eccc4e67ab52d`. Both copies have
SHA-256
`464a8a95159d5e26ad24a702107aec86698935deeb034e02c9fe51e55472d75b`.
`cmp` reported an exact match. The upstream Apache-2.0 text is retained as
`licenses/kimodo-LICENSE.txt`; `licenses/kimodo-ATTRIBUTION.txt` records the
provenance in the built package.

## Source-Code Provenance Check

The new integration calls the installed SOMA-X public APIs (`SOMALayer`,
`PoseInversion`, and transform helpers) instead of bundling the SOMA-X package.
A normalized line-level comparison of the added conversion modules against
the installed SOMA-X and Kimodo Python source found no substantive contiguous
copy. The longest match was seven argument-forwarding lines in the
`PoseInversion` call; other matches were common dictionary fields or one to
three generic lines.

All Python files added by the branch contain an Apache-2.0 SPDX identifier.
Files with NVIDIA copyright headers follow the existing repository convention,
including the AI Sapiens files already present on `origin/main`.

## Package Boundary

The base installation remains Apache-2.0 code and does not import SOMA-X,
Torch, or `smplx` unless the optional conversion path is requested. The
`soma-x` extra intentionally resolves `py-soma-x[smpl]==0.2.1`, which in turn
installs `smplx`. This does not copy `smplx` into the Soma Retargeter wheel,
but users of the extra must comply with the installed package and identity
model licenses. The README and integration guide now state this boundary
explicitly.

## DCO Process Note

`CONTRIBUTING.md` requires signed-off commits for submissions to the NVIDIA
upstream project. Existing branch commits `0a4d57a` and `133cd90` do not contain
`Signed-off-by` trailers. This does not change the code license, but those
commits do not satisfy the documented upstream contribution process as-is.
The current audit commit is created with `git commit --signoff`. Rewriting the
two existing commits was not performed because the request did not authorize
history rewriting.

## Verification

Commands executed in `crazy_spence_mounted:/soma-retargeter`:

```bash
PYTHONPATH=/soma-retargeter /.venv/bin/python -m unittest discover -s tests -v
/.venv/bin/python -m pip wheel . --no-deps --no-build-isolation \
  --wheel-dir /tmp/soma-license-wheel-20260825_114513
/.venv/bin/python -m pip check
cmp /kimodo/LICENSE licenses/kimodo-LICENSE.txt
cmp /kimodo/kimodo/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p \
  soma_retargeter/configs/soma/standard_t_pose_global_offsets_rots.p
```

Results:

- All 64 unit tests passed.
- The wheel built successfully as
  `soma_retargeter-0.1.0-py3-none-any.whl`, SHA-256
  `96bdd063c76f4b4ce7a46e637f2675a94eb875bb508e2fb68a9a87a478b95f7b`.
- The wheel contains the SOMA77 offsets asset, the exact Kimodo license, and
  the Kimodo attribution record.
- Wheel inspection found no unexpected `.pkl` or `.npz` identity-model/data
  artifacts.
- `pip check` reported no broken requirements.
- The packaged asset and Kimodo source asset are byte-identical and have the
  expected SHA-256.
- The copied Kimodo license is byte-identical to `/kimodo/LICENSE`.
- Static inspection found no SMPL-family identity model under `app/`,
  `soma_retargeter/`, `tools/`, `tests/`, `docs/`, or `licenses/`.

## Conclusion

No blocking redistribution conflict was found for the base Soma Retargeter
wheel or the bundled Kimodo asset after adding the license and attribution
records. The optional SMPL conversion path is not license-neutral: installing
the `soma-x` extra brings in `smplx`, and both that software and user-supplied
SMPL-family model files require compliance with their separate terms. This
restriction is now disclosed in the README and integration guide rather than
being represented as part of the Apache-2.0 grant.
