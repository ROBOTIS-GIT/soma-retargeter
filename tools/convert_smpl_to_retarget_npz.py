#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Canonical SMPL/SMPL-H/SMPL-X to SOMA77 conversion CLI."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.convert_smplx_to_retarget_npz import *  # noqa: F403
from tools.convert_smplx_to_retarget_npz import main as _compat_main


if __name__ == "__main__":
    _compat_main(default_model_type="auto")
