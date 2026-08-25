# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from soma_retargeter.assets import kimodo_npz


class KimodoNpzOffsetsTests(unittest.TestCase):
    def test_packaged_default_offsets_are_valid_rotations(self) -> None:
        expected = (
            Path(kimodo_npz.__file__).resolve().parents[1]
            / "configs"
            / "soma"
            / "standard_t_pose_global_offsets_rots.p"
        )

        resolved = kimodo_npz.find_default_offsets_path()

        self.assertEqual(resolved, expected)
        offsets = kimodo_npz.load_global_offsets(resolved)
        self.assertEqual(offsets.shape, (77, 3, 3))
        np.testing.assert_allclose(
            np.swapaxes(offsets, -1, -2) @ offsets,
            np.broadcast_to(np.eye(3), offsets.shape),
            atol=1e-5,
        )
        np.testing.assert_allclose(np.linalg.det(offsets), 1.0, atol=1e-5)

    def test_npz_to_bvh_uses_packaged_offsets_without_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_npz = root / "motion.npz"
            output_bvh = root / "motion.bvh"
            identity = np.eye(3, dtype=np.float32)
            local_rot_mats = np.broadcast_to(
                identity,
                (2, 77, 3, 3),
            ).copy()
            np.savez(
                input_npz,
                local_rot_mats=local_rot_mats,
                root_positions=np.zeros((2, 3), dtype=np.float32),
                fps=np.float32(30.0),
            )

            result = kimodo_npz.convert_npz_to_bvh(
                input_npz,
                output_bvh,
                fps=30.0,
            )

            self.assertTrue(output_bvh.exists())
            self.assertEqual(result["frames"], 2)
            self.assertEqual(result["fps"], 30.0)
            self.assertEqual(
                Path(result["offsets"]),
                Path(kimodo_npz.__file__).resolve().parents[1]
                / "configs"
                / "soma"
                / "standard_t_pose_global_offsets_rots.p",
            )


if __name__ == "__main__":
    unittest.main()
