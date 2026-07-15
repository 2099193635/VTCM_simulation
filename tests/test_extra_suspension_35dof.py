from types import SimpleNamespace

import numpy as np
import pytest

from configs.parameters import (
    Antiyawer_parameters,
    ExtraForceElements_parameters,
    VehicleParams,
)
from configs.topology import SystemTopology
from physics_modules.equation_of_motion import GeneralForceAssembler
from physics_modules.solver import DynamicSolver
from physics_modules.suspension import SuspensionSystem


def _extra_params():
    return SimpleNamespace(
        Lpvdx_Bogie=1.5,
        dpvd=0.8,
        dnode=0.7,
        Ht_node=0.4,
        Lnodex_Bogie=0.9,
        Lnode_axlebox=0.3,
        dsvd=1.2,
        Lsvd_Car=8.0,
        Ht_sld=0.2,
        Hc_sld=1.0,
        Lsld_Car_L=np.array([9.1, 8.9]),
        Lsld_Car_R=np.array([8.9, 9.1]),
        Lsld_Bogie=0.1,
    )


def _zero_extra_forces():
    return {
        'Fpvdz_L': np.zeros(4), 'Fpvdz_R': np.zeros(4),
        'Fnodex_L': np.zeros(4), 'Fnodex_R': np.zeros(4),
        'Fnodey_L': np.zeros(4), 'Fnodey_R': np.zeros(4),
        'Fnodez_L': np.zeros(4), 'Fnodez_R': np.zeros(4),
        'Fsvdz_L': np.zeros(2), 'Fsvdz_R': np.zeros(2),
        'Fsldy_L': np.zeros(2), 'Fsldy_R': np.zeros(2),
    }


def _assembler():
    assembler = GeneralForceAssembler.__new__(GeneralForceAssembler)
    assembler.ep = _extra_params()
    return assembler


def test_curve_inputs_preserve_signed_radius_and_zero_curvature():
    geometry = {
        'K': np.array([[10.0, 20.0, 30.0, 40.0, 0.002, -0.004, 0.0]]),
        'dK': np.array([[0.0, 0.0, 0.0, 0.0, 1e-5, -2e-5, 0.0]]),
    }

    radius, rate = DynamicSolver._suspension_curve_inputs(geometry, 0)

    np.testing.assert_allclose(radius[:2], [500.0, -250.0])
    assert np.isinf(radius[2])
    np.testing.assert_allclose(rate, [1e-5, -2e-5, 0.0])
    assert not np.isnan(radius).any()


@pytest.mark.parametrize(
    ('force_name', 'element_index', 'translational_dofs'),
    [
        ('Fpvdz_L', 0, (6, 16)),
        ('Fnodey_L', 0, (5, 15)),
        ('Fsvdz_L', 0, (1, 6)),
        ('Fsldy_L', 0, (0, 5)),
    ],
)
def test_internal_extra_elements_balance_translational_force(
    force_name, element_index, translational_dofs
):
    forces = _zero_extra_forces()
    forces[force_name][element_index] = 123.0

    q = _assembler().assemble_extra_vehicle_forces(forces)

    first, second = translational_dofs
    assert q[first] == pytest.approx(-q[second])
    assert abs(q[first]) == pytest.approx(123.0)


def test_pvd_and_node_moments_use_configured_geometry():
    forces = _zero_extra_forces()
    forces['Fpvdz_L'][0] = 10.0
    forces['Fnodex_L'][0] = 20.0

    q = _assembler().assemble_extra_vehicle_forces(forces)
    ep = _extra_params()

    assert q[7] == pytest.approx(10.0 * ep.dpvd)
    assert q[17] == pytest.approx(-10.0 * ep.dpvd)
    assert q[8] == pytest.approx(10.0 * ep.Lpvdx_Bogie - 20.0 * ep.Ht_node)
    assert q[9] == pytest.approx(-20.0 * ep.dnode)
    assert q[19] == pytest.approx(20.0 * ep.dnode)


def test_extra_force_switch_and_optional_axlebox_interface():
    vehicle = VehicleParams()
    antiyaw = Antiyawer_parameters()
    extra = ExtraForceElements_parameters(Lc=vehicle.Lc)
    suspension = SuspensionSystem(vehicle, antiyaw, extra)
    topology = SystemTopology(Nt=1, Nsub=1, NV=1, NL=1, NT=1)
    displacement = np.zeros(topology.Fnum_Total)
    velocity = np.zeros_like(displacement)
    displacement[:35] = np.linspace(0.0, 0.01, 35)
    velocity[:35] = np.linspace(0.0, 0.02, 35)
    state = topology.extract_state(displacement, velocity, Vc=20.0)
    keys = ('Fpvdz_L', 'Fnodex_L', 'Fnodey_L', 'Fnodez_L',
            'Fsvdz_L', 'Fsldy_L')

    disabled = suspension.compute_forces(state, include_extra=False)
    enabled_default = suspension.compute_forces(state, include_extra=True)
    enabled_zero_axlebox = suspension.compute_forces(
        state,
        include_extra=True,
        axlebox_state={
            'spin_L': np.zeros(4), 'spin_R': np.zeros(4),
            'spin_vel_L': np.zeros(4), 'spin_vel_R': np.zeros(4),
        },
    )

    assert all(np.allclose(disabled[key], 0.0) for key in keys)
    assert all(np.allclose(enabled_default[key], enabled_zero_axlebox[key]) for key in keys)
    groups = (
        ('Fpvdz_L', 'Fpvdz_R'),
        ('Fnodex_L', 'Fnodey_L', 'Fnodez_L'),
        ('Fsvdz_L', 'Fsvdz_R'),
        ('Fsldy_L', 'Fsldy_R'),
    )
    assert all(
        sum(np.linalg.norm(enabled_default[key]) for key in group) > 0.0
        for group in groups
    )

    with pytest.raises(ValueError, match='must have shape'):
        suspension.compute_forces(
            state,
            include_extra=True,
            axlebox_state={'spin_L': np.zeros(3)},
        )