import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "multilayer_AlSc",
]


def multilayer_AlSc() -> optika.materials.MultilayerMirror:
    r"""
    A proposed coating design for ESIS II which uses :math:`\text{Al/Sc}`
    layers

    Examples
    --------

    Plot the efficiency of the coating across the EUV range

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        import esis

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Initialize the ESIS primary mirror material
        material = esis.flights.flight_02.optics.materials.multilayer_AlSc()

        # Compute the reflectivity of the primary mirror
        reflectivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, reflectivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit})");
        ax.set_ylabel("reflectivity");
    """
    n = 10
    d = 257 * u.AA
    gamma = 0.5
    t_Al = gamma * d
    t_Sc = (1 - gamma) * d
    return optika.materials.MultilayerMirror(
        material_layers=na.ScalarArray(np.array(n * ["Al", "Sc"]), axes="layer"),
        material_substrate="SiO2",
        thickness_layers=na.ScalarArray(u.Quantity(n * [t_Al, t_Sc]), axes="layer"),
        axis_layers="layer",
        profile_interface=optika.materials.profiles.ErfInterfaceProfile(7 * u.AA),
    )
