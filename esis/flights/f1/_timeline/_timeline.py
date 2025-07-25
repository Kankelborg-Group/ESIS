import astropy.units as u
import esis

__all__ = [
    "timeline",
]


def timeline() -> esis.nsroc.Timeline:
    """
    The timeline of ESIS mission events provided by NSROC.
    """
    return esis.nsroc.Timeline(
        timedelta_esis_start=0.1 * u.s,
        timedelta_rail_release=0.6 * u.s,
        timedelta_terrier_burnout=6.2 * u.s,
        timedelta_blackbrant_ignition=16.0 * u.s,
        timedelta_canard_decouple=20.0 * u.s,
        timedelta_blackbrant_burnout=43.5 * u.s,
        timedelta_despin=62.0 * u.s,
        timedelta_payload_separation=66.0 * u.s,
        timedelta_sparcs_enable=69.5 * u.s,
        timedelta_shutter_open=73.0 * u.s,
        timedelta_nosecone_eject=81.0 * u.s,
        timedelta_sparcs_finemode=119.2 * u.s,
        timedelta_sparcs_rlg_enable=124.2 * u.s,
        timedelta_sparcs_rlg_disable=431.0 * u.s,
        timedelta_shutter_close=433.0 * u.s,
        timedelta_sparcs_spinup=439.0 * u.s,
        timedelta_sparcs_vent=455.0 * u.s,
        timedelta_ballistic_impact=526.5 * u.s,
        timedelta_sparcs_disable=555.0 * u.s,
        timedelta_parachute_deploy=569.8 * u.s,
        timedelta_payload_impact=849.9 * u.s,
    )
