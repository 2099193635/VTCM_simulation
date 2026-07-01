import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.build_joint_inverse_dataset import build_dataset, parse_args
from physics_modules.structure_defects import StructureDefectManager
from physics_modules.substructure import SubstructureDynamics


class StructureDefectTests(unittest.TestCase):
    def make_substructure(self):
        fp = SimpleNamespace(Kkjv=10.0, Ckjv=2.0, Kkjh=5.0, Ckjh=1.0)
        rp = SimpleNamespace(bb=0.1, aa=0.2)
        sp = SimpleNamespace(d=0.5, Kbv=100.0, Cbv=10.0, Kbh=20.0, Cbh=2.0, Kw=1.0, Cw=0.1, Kfv=3.0, Cfv=0.3)
        return SubstructureDynamics(fp, rp, sp)

    def make_state(self, n=3):
        zeros = np.zeros(n)
        return {
            'XSleeper_Z': zeros.copy(),
            'XSleeper_Y': zeros.copy(),
            'XSleeper_Roll': zeros.copy(),
            'VSleeper_Z': zeros.copy(),
            'VSleeper_Y': zeros.copy(),
            'VSleeper_Roll': zeros.copy(),
            'XSubgrade_L': zeros.copy(),
            'VSubgrade_L': zeros.copy(),
            'XSubgrade_R': zeros.copy(),
            'VSubgrade_R': zeros.copy(),
        }

    def make_rail_state(self, n=3):
        ones = np.ones(n)
        zeros = np.zeros(n)
        return {
            'RailF_Zdis_L': ones.copy(),
            'RailF_Tdis_L': zeros.copy(),
            'RailF_Zvel_L': zeros.copy(),
            'RailF_Tvel_L': zeros.copy(),
            'RailF_Ldis_L': ones.copy(),
            'RailF_Lvel_L': zeros.copy(),
            'RailF_Zdis_R': ones.copy(),
            'RailF_Tdis_R': zeros.copy(),
            'RailF_Zvel_R': zeros.copy(),
            'RailF_Tvel_R': zeros.copy(),
            'RailF_Ldis_R': ones.copy(),
            'RailF_Lvel_R': zeros.copy(),
        }

    def test_failed_fastener_zeroes_selected_node_force(self):
        sub = self.make_substructure()
        ss = self.make_state()
        rs = self.make_rail_state()
        defects = {
            'fastener_kv_factor_L': np.array([1.0, 0.0, 1.0]),
            'fastener_cv_factor_L': np.array([1.0, 0.0, 1.0]),
            'fastener_kh_factor_L': np.array([1.0, 0.0, 1.0]),
            'fastener_ch_factor_L': np.array([1.0, 0.0, 1.0]),
        }

        forces = sub.compute_fastener_forces(rs, ss, defects=defects)

        self.assertGreater(abs(forces['FV_L'][0]), 0.0)
        self.assertEqual(forces['FV_L'][1], 0.0)
        self.assertEqual(forces['FL_L'][1], 0.0)
        self.assertGreater(abs(forces['FV_L'][2]), 0.0)

    def test_sleeper_void_gap_uses_one_sided_contact(self):
        sub = self.make_substructure()
        ss = self.make_state(n=1)
        ss['XSleeper_Z'][0] = 0.0005
        defects = {
            'void_active_L': np.array([True]),
            'void_gap_L': np.array([0.001]),
        }

        open_gap = sub.compute_subrail_forces(ss, ss, defects=defects)
        self.assertEqual(open_gap['FLsV'][0], 0.0)
        self.assertFalse(open_gap['Void_contact_L'][0])

        ss['XSleeper_Z'][0] = 0.002
        closed_gap = sub.compute_subrail_forces(ss, ss, defects=defects)
        self.assertAlmostEqual(closed_gap['FLsV'][0], 0.1)
        self.assertTrue(closed_gap['Void_contact_L'][0])

    def test_absolute_defect_maps_into_moving_window(self):
        ip = SimpleNamespace(S0_mileage=1000.0, Nsub=4, Lkj=0.6, Cord_fastener=np.linspace(0.0, 2.4, 5))
        sw = {'window_start_abs_m': np.array([900.0, 1000.0])}
        records = StructureDefectManager._parse_records([
            {'type': 'fastener_failure', 'abs_mileage_m': 1001.2, 'count': 1, 'side': 'left'}
        ], ip)
        manager = StructureDefectManager(ip, sw, records, enabled=True)

        step = manager.get_step_defects(1)

        self.assertTrue(step['fastener_active_L'][2])
        self.assertEqual(np.count_nonzero(step['fastener_active_L']), 1)

    def test_ballast_stiffness_field_is_spatial_label(self):
        ip = SimpleNamespace(S0_mileage=1000.0, Nsub=10, Lkj=0.6, Cord_fastener=np.linspace(0.0, 6.0, 11))
        records = StructureDefectManager._parse_records([
            {
                'type': 'ballast_condition',
                'abs_mileage_m': 1001.2,
                'count': 2,
                'side': 'both',
                'directions': 'vertical',
                'stiffness_factor_eta_k': 0.2,
                'damping_factor_eta_c': 1.0,
            }
        ], ip)
        manager = StructureDefectManager(ip, {}, records, enabled=True)

        field = manager.ballast_stiffness_field(np.array([1000.6, 1001.2, 1001.8, 1002.4]))

        np.testing.assert_allclose(field['eta_k_L'], np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32))
        np.testing.assert_allclose(field['eta_k_R'], np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32))
        self.assertEqual(int(np.count_nonzero(field['mask_L'])), 2)

    def test_joint_inverse_dataset_builder_from_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demo_run" / "files"
            root.mkdir(parents=True)
            n = 80
            rel = np.arange(n, dtype=np.float64) * 0.25
            abs_s = 1000.0 + rel
            a = np.zeros((n, 35), dtype=np.float32)
            a[:, 1] = np.sin(rel).astype(np.float32)
            eta_l = np.ones(n, dtype=np.float32)
            eta_r = np.ones(n, dtype=np.float32)
            eta_l[20:40] = 0.2
            eta_r[20:40] = 0.2
            np.savez(
                root / "simulation_result.npz",
                A=a,
                dt=np.array(0.01),
                Track_rel_mileage_m=rel,
                Track_abs_mileage_m=abs_s,
                Irre_bz_L_ref=np.zeros(n, dtype=np.float32),
                Irre_bz_R_ref=np.zeros(n, dtype=np.float32),
                Irre_by_L_ref=np.zeros(n, dtype=np.float32),
                Irre_by_R_ref=np.zeros(n, dtype=np.float32),
                Stiffness_eta_k_L_ref=eta_l,
                Stiffness_eta_k_R_ref=eta_r,
                Stiffness_irregularity_mask_L=(eta_l != 1.0).astype(np.int8),
                Stiffness_irregularity_mask_R=(eta_r != 1.0).astype(np.int8),
            )

            args = parse_args([])
            args.result_roots = [str(Path(tmp))]
            args.ds_m = 0.25
            args.window_m = 8.0
            args.stride_m = 4.0
            args.include_force = False
            args.min_valid_fraction = 0.0
            dataset = build_dataset(args)

            self.assertEqual(dataset['obs'].shape[1:], (32, 10))
            self.assertEqual(dataset['geom_irregularity'].shape[-1], 4)
            self.assertEqual(dataset['stiffness_irregularity'].shape[-1], 2)
            self.assertTrue(np.any(dataset['stiffness_irregularity'] < 1.0))


    def test_subgrade_condition_maps_to_kfv_cfv_only(self):
        ip = SimpleNamespace(S0_mileage=1000.0, Nsub=2, Lkj=0.5, Cord_fastener=np.array([0.0, 0.5, 1.0]))
        records = StructureDefectManager._parse_records([
            {
                'type': 'subgrade_void',
                'start_m': 0.5,
                'count': 1,
                'side': 'left',
                'eta_k': 0.25,
                'eta_c': 0.5,
            }
        ], ip)
        manager = StructureDefectManager(ip, {}, records, enabled=True)

        defects = manager.get_step_defects(0)

        np.testing.assert_allclose(defects['subgrade_kv_factor_L'], [1.0, 0.25, 1.0])
        np.testing.assert_allclose(defects['subgrade_cv_factor_L'], [1.0, 0.5, 1.0])
        np.testing.assert_allclose(defects['ballast_kv_factor_L'], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(defects['fastener_kv_factor_L'], [1.0, 1.0, 1.0])
        self.assertTrue(defects['subgrade_active_L'][1])

    def test_subgrade_condition_changes_foundation_force_only(self):
        sub = self.make_substructure()
        ss = self.make_state(n=1)
        ss['XSubgrade_L'][0] = 2.0
        ss['VSubgrade_L'][0] = 4.0
        defects = {
            'subgrade_kv_factor_L': np.array([0.5]),
            'subgrade_cv_factor_L': np.array([0.25]),
        }

        baseline = sub.compute_subrail_forces(ss, ss, defects={})
        weakened = sub.compute_subrail_forces(ss, ss, defects=defects)

        self.assertAlmostEqual(baseline['FLbf'][0], 7.2)
        self.assertAlmostEqual(weakened['FLbf'][0], 3.3)
        self.assertAlmostEqual(baseline['FLsV'][0], weakened['FLsV'][0])

if __name__ == '__main__':
    unittest.main()
