import astropy.units as u

__all__ = [
    "esis_exp_launch",
    "rail_release",
    "terrier_burnout",
    "black_brant_ignition",
    "canard_decouple",
    "black_brant_burnout",
    "despin",
    "payload_separation",
    "sparcs_enable",
    "shutter_door_open",
    "nosecone_eject",
    "sparcs_rlg_enable",
    "sparcs_rlg_disable",
    "shutter_door_close",
    "sparcs_spin_up",
    "sparcs_vent",
    "ballistic_impact",
    "sparcs_disable",
    "parachute_deploy",
    "payload_impact",
]

esis_exp_launch = 0.1 * u.s
rail_release = 0.6 * u.s
terrier_burnout = 6.2 * u.s
black_brant_ignition = 16.0 * u.s
canard_decouple = 20.0 * u.s
black_brant_burnout = 43.5 * u.s
despin = 62.0 * u.s
payload_separation = 66.0 * u.s
sparcs_enable = 69.5 * u.s
shutter_door_open = 73.0 * u.s
nosecone_eject = 81.0 * u.s
sparcs_fine_mode_stable = 119.2 * u.s
sparcs_rlg_enable = 124.2 * u.s
sparcs_rlg_disable = 431.0 * u.s
shutter_door_close = 433.0 * u.s
sparcs_spin_up = 439.0 * u.s
sparcs_vent = 455.0 * u.s
ballistic_impact = 526.5 * u.s
sparcs_disable = 555.0 * u.s
parachute_deploy = 569.8 * u.s
payload_impact = 849.9 * u.s
