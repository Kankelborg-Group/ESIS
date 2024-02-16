import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "multilayer_design",
]


def multilayer_design() -> optika.materials.MultilayerMirror:
    """
    The as-designed multilayer coating for the ESIS primary mirror.

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
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=101) * u.AA

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Initialize the ESIS primary mirror material
        material = esis.flights.flight_01.optics.primary_mirrors.materials.multilayer_design()

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
    return optika.materials.MultilayerMirror(
        material_layers=na.ScalarArray(np.array(["SiO2", "SiC", "Cr"]), axes="layer"),
        material_substrate="SiO2",
        thickness_layers=na.ScalarArray([1, 25, 5] * u.nm, axes="layer"),
        axis_layers="layer",
        profile_interface=optika.materials.profiles.ErfInterfaceProfile(2 * u.nm),
    )
