from types import SimpleNamespace

import numpy as np
import pytest

from configs.topology import SystemTopology
from physics_modules.solver import DynamicSolver, SystemDynamics
from physics_modules.wheel_rail_contact import ContactMechanics


def _vehicle_params():
    return SimpleNamespace(
        Mc=1.0,
        MB=1.0,
        Mt=1.0,
        Mw=1.0,
        Jcx=1.0,
        Jcy=1.0,
        Jcz=1.0,
        Jtx=1.0,
        Jty=1.0,
        Jtz=1.0,
        Jwx=1.0,
        Jwy=1.0,
        Jwz=1.0,
    )


def test_creep_force_is_nonzero_and_friction_limited():
    normal_force = 70_000.0
    fx, fy, mz = ContactMechanics.calc_creep_forces(
        ksix=0.001,
        ksiy=0.002,
        ksisp=0.0,
        Rw=0.46,
        rw=0.5,
        rr=0.3,
        N=normal_force,
    )

    assert np.isfinite([fx, fy, mz]).all()
    assert abs(fx) + abs(fy) > 0.0
    assert np.hypot(fx, fy) <= 0.3 * normal_force + 1e-6


def test_sleeper_roll_uses_rotational_inertia():
    integration = SimpleNamespace(Nsub=2)
    subrail = SimpleNamespace(Ms=237.0, Mb=10.0, Js=123.4375)
    modes = SimpleNamespace(NV=1, NL=1, NT=1)
    dynamics = SystemDynamics(
        _vehicle_params(),
        integration,
        subrail,
        modes,
        SimpleNamespace(),
    )

    node_count = integration.Nsub + 1
    sleeper_start = 35 + 2 * (modes.NV + modes.NL + modes.NT)
    sleeper_mass = dynamics.Mass_FULL[
        sleeper_start:sleeper_start + 3 * node_count
    ]

    np.testing.assert_allclose(sleeper_mass[:2 * node_count], subrail.Ms)
    np.testing.assert_allclose(sleeper_mass[2 * node_count:], subrail.Js)


def test_axlebox_lock_rejects_topology_without_axlebox_dofs():
    topology = SystemTopology(Nt=2, Nsub=2, NV=1, NL=1, NT=1)

    with pytest.raises(ValueError, match="no axlebox DOFs"):
        DynamicSolver(
            topology,
            SimpleNamespace(),
            switch_lock_veh_non_z="Off",
            switch_lock_axlebox="On",
            switch_lock_substructure="Off",
        )

