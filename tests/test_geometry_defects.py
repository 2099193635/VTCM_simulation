import unittest

import numpy as np

from defect_injector.geometry_defects import build_geometry_defect_profile, normalize_geometry_defects
from defect_injector.irregularity import Irregularity


class GeometryDefectTests(unittest.TestCase):
    def test_short_wave_defects_generate_channel_profiles(self):
        specs = normalize_geometry_defects([
            {
                'type': 'weld_joint',
                'start_m': 10.0,
                'length_m': 1.0,
                'amplitude_mm': 2.0,
                'channel': 'vertical',
                'side': 'left',
            },
            {
                'type': 'corrugation',
                'start_m': 20.0,
                'length_m': 3.0,
                'amplitude_mm': 0.5,
                'wavelength_m': 0.3,
                'channel': 'lateral',
                'side': 'right',
            },
        ])
        distance = np.linspace(0.0, 30.0, 601)

        profile, metadata = build_geometry_defect_profile(distance, specs)

        self.assertEqual(len(metadata), 2)
        self.assertGreater(np.max(np.abs(profile['VL'])), 0.0)
        self.assertEqual(np.max(np.abs(profile['VR'])), 0.0)
        self.assertGreater(np.max(np.abs(profile['LR'])), 0.0)
        self.assertEqual(np.max(np.abs(profile['LL'])), 0.0)

    def test_medium_long_wave_placeholders_parse_but_return_zero_profile(self):
        specs = normalize_geometry_defects([
            {'type': 'subgrade_settlement', 'start_m': 5.0, 'length_m': 20.0, 'amplitude_mm': 8.0},
            {'type': 'frost_heave', 'start_m': 50.0, 'length_m': 10.0, 'amplitude_mm': 5.0},
        ])
        distance = np.linspace(0.0, 100.0, 501)

        profile, metadata = build_geometry_defect_profile(distance, specs)

        self.assertEqual(len(metadata), 2)
        self.assertTrue(all(not item['implemented'] for item in metadata))
        for arr in profile.values():
            np.testing.assert_allclose(arr, 0.0)

    def test_external_import_branch_does_not_apply_geometry_defects(self):
        ir = Irregularity(
            Lc=1.0,
            Lt=1.0,
            Vc=10.0,
            Tstep=0.1,
            Tz=1.0,
            Nt=10,
            type='外部导入',
            geometry_defect_specs=[{'type': 'weld_joint', 'start_m': 0.0, 'amplitude_mm': 5.0}],
        )
        ir._build_zero_geometry_defect_profile(8)

        self.assertEqual(len(ir.geometry_defect_profile_full['VL']), 8)
        np.testing.assert_allclose(ir.geometry_defect_profile_full['VL'], 0.0)


if __name__ == '__main__':
    unittest.main()