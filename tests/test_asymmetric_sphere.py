"""Regression check for asymmetric sphere construction."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from membrane_curved_bumpy import bumpy


class AsymmetricSphereTest(unittest.TestCase):
    """Ensure asymmetric pivotal plane offsets map to different leaflet areas."""

    def test_leaflet_area_ratio_matches_offsets(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        template_path = base_dir / "membrane_curved_bumpy" / "flat.pdb"

        template = bumpy.Molecules(infile=str(template_path))

        r_sphere = 60.0
        zo = (15.0, 10.0)

        lateral_requirements = bumpy.shapes.semisphere.dimension_requirements(r_sphere)
        template_dims = np.array(template.boxdims[0:2], dtype=float)
        mult_factor = np.ceil(lateral_requirements / template_dims).astype(int)
        template.duplicate_laterally(int(mult_factor[0]), int(mult_factor[1]))

        sphere = bumpy.shapes.sphere.gen_shape(template, list(zo), r_sphere)

        center = sphere.coords.mean(axis=0)
        distances = np.linalg.norm(sphere.coords - center, axis=1)
        top_mask = sphere.metadata.leaflets.astype(bool)
        bottom_mask = np.invert(top_mask)

        top_radius = distances[top_mask].mean()
        bottom_radius = distances[bottom_mask].mean()

        expected_top_radius = r_sphere + zo[0]
        expected_bottom_radius = r_sphere - zo[1]
        expected_ratio = (expected_top_radius ** 2) / (expected_bottom_radius ** 2)
        observed_ratio = (top_radius ** 2) / (bottom_radius ** 2)

        self.assertGreater(top_radius, bottom_radius)
        self.assertAlmostEqual(top_radius, expected_top_radius, delta=2.5)
        self.assertAlmostEqual(bottom_radius, expected_bottom_radius, delta=2.5)
        self.assertAlmostEqual(observed_ratio, expected_ratio, delta=0.05 * expected_ratio)


if __name__ == "__main__":
    unittest.main()
