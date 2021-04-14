import dataclasses
import pathlib
import astropy.units as u
import kgpy
import kgpy.nsroc
import esis.data

__all__ = [
    'trajectory_file',
    'trajectory',
    'timeline',
]

trajectory_file = pathlib.Path(__file__).parent / '36320_Trajectory.txt'


def trajectory() -> kgpy.nsroc.Trajectory:
    return kgpy.nsroc.Trajectory.from_nsroc_csv(csv_file=trajectory_file)


def timeline() -> esis.data.nsroc.Timeline:
    tl = esis.data.nsroc.Timeline()
    tl.esis_exp_launch.time_mission = 0.1 * u.s
    tl.rail_release.time_mission = 0.6 * u.s
    tl.terrier_burnout.time_mission = 6.2 * u.s
    tl.black_brant_ignition.time_mission = 16.0 * u.s
    tl.canard_decouple.time_mission = 20.0 * u.s
    tl.black_brant_burnout.time_mission = 43.5 * u.s
    tl.despin.time_mission = 62.0 * u.s
    tl.payload_separation.time_mission = 66.0 * u.s
    tl.sparcs_enable.time_mission = 69.5 * u.s
    tl.shutter_door_open.time_mission = 73.0 * u.s
    tl.nosecone_eject.time_mission = 81.0 * u.s
    tl.sparcs_fine_mode_stable.time_mission = 119.2 * u.s
    tl.sparcs_rlg_enable.time_mission = 124.2 * u.s
    tl.sparcs_rlg_disable.time_mission = 431.0 * u.s
    tl.shutter_door_close.time_mission = 433.0 * u.s
    tl.sparcs_spin_up.time_mission = 439.0 * u.s
    tl.sparcs_vent.time_mission = 455.0 * u.s
    tl.ballistic_impact.time_mission = 526.5 * u.s
    tl.sparcs_disable.time_mission = 555.0 * u.s
    tl.parachute_deploy.time_mission = 569.8 * u.s
    tl.payload_impact.time_mission = 849.9 * u.s
    return tl
