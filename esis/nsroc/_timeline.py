import dataclasses
import astropy.units as u
import optika

__all__ = [
    "Timeline",
]


@dataclasses.dataclass
class Timeline(
    optika.mixins.Printable,
):
    """
    A Python representation of the NSROC timeline for the ESIS mission.
    """

    timedelta_esis_start: u.Quantity
    """
    The amount of time between mission start and the start of the ESIS exposure
    sequence.
    """

    timedelta_rail_release: u.Quantity
    """
    The amount of time between mission start and the moment that the vehicle
    clears the launch rail.
    """

    timedelta_terrier_burnout: u.Quantity
    """
    The amount of time between mission start and the burnout of the Terrier 
    first stage.
    """

    timedelta_blackbrant_ignition: u.Quantity
    """
    The amount of time between mission start and the ignition of the 
    Black Brant second stage.
    """

    timedelta_canard_decouple: u.Quantity
    """
    The amount of time between mission start and the moment that the S-19
    guidance system releases the canards.
    """

    timedelta_blackbrant_burnout: u.Quantity
    """
    The amount of time between mission start and the burnout of the Black Brant
    second stage.
    """

    timedelta_despin: u.Quantity
    """
    The amount of time between mission start and the start of the despin
    sequence.
    """

    timedelta_payload_separation: u.Quantity
    """
    The amount of time between mission start and the payload separation from
    the rest of the vehicle.
    """

    timedelta_sparcs_enable: u.Quantity
    """
    The amount of time between mission start and the moment that the SPARCS
    pointing system is enabled.
    """

    timedelta_shutter_open: u.Quantity
    """
    The amount of time between mission start and the opening of the payload
    shutter door.
    """

    timedelta_nosecone_eject: u.Quantity
    """
    The amount of time between mission start and the ejection from the nosecone
    on the top of the payload.
    """

    timedelta_sparcs_finemode: u.Quantity
    """
    The amount of time between mission start and the acquisition of fine-pointing
    mode with SPARCS.
    """

    timedelta_sparcs_rlg_enable: u.Quantity
    """
    The amount of time between mission start and the enabling of the ring-laser
    gyroscope to control the roll.
    """

    timedelta_sparcs_rlg_disable: u.Quantity
    """
    The amount of time between mission start and the disabling of the ring-laser
    gyroscope.
    """

    timedelta_shutter_close: u.Quantity
    """
    The amount of time between mission start and the closing of the payload
    shutter door.
    """

    timedelta_sparcs_spinup: u.Quantity
    """
    The amount of time between mission start and the spin up of the payload
    to prevent overheating on re-entry.
    """

    timedelta_sparcs_vent: u.Quantity
    """
    The amount of time between mission start and the venting of the SPARCS
    system.
    """

    timedelta_ballistic_impact: u.Quantity
    """
    The amount of time between mission start and ballistic impact of the
    payload if it weren't caught by the parachute.
    """

    timedelta_sparcs_disable: u.Quantity
    """
    The amount of time between mission start and the moment that the SPARCS
    pointing system is disabled.
    """

    timedelta_parachute_deploy: u.Quantity
    """
    The amount of time between mission start and deployment of the parachute.
    """

    timedelta_payload_impact: u.Quantity
    """
    The amount of time between mission start and the payload impact on the
    chute.
    """
