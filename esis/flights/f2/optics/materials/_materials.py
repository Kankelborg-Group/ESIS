import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "multilayer_AlSc",
    "multilayer_SiSc",
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

        # Initialize the multilayer material
        material = esis.flights.f2.optics.materials.multilayer_AlSc()

        # Compute the reflectivity of the multilayer material
        reflectivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, reflectivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
    """
    n = 10
    d = 257 * u.AA
    gamma = 0.5
    t_Al = gamma * d
    t_Sc = (1 - gamma) * d
    t_Al2O3 = 1 * u.nm
    return optika.materials.MultilayerMirror(
        material_layers=na.ScalarArray(
            ndarray=np.array(["Al2O3"] + n * ["Al", "Sc"]),
            axes="layer",
        ),
        material_substrate="SiO2",
        thickness_layers=na.ScalarArray(
            ndarray=u.Quantity([t_Al2O3] + n * [t_Al, t_Sc]),
            axes="layer",
        ),
        axis_layers="layer",
        profile_interface=optika.materials.profiles.ErfInterfaceProfile(7 * u.AA),
    )


def multilayer_SiSc() -> optika.materials.MultilayerMirror:
    r"""
    A proposed coating design for ESIS II which uses :math:`\text{Si/Sc}`
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

        # Initialize the multilayer material
        material = esis.flights.f2.optics.materials.multilayer_SiSc()

        # Compute the reflectivity of the multilayer
        reflectivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, reflectivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
    """
    n = 5
    d = 260 * u.AA
    gamma = 0.5
    t_Si = gamma * d
    t_Sc = (1 - gamma) * d
    t_SiO2 = 1 * u.nm
    return optika.materials.MultilayerMirror(
        material_layers=na.ScalarArray(
            ndarray=np.array(["SiO2"] + n * ["Si", "Sc"]),
            axes="layer",
        ),
        material_substrate="SiO2",
        thickness_layers=na.ScalarArray(
            ndarray=u.Quantity([t_SiO2] + n * [t_Si, t_Sc]),
            axes="layer",
        ),
        axis_layers="layer",
        profile_interface=optika.materials.profiles.ErfInterfaceProfile(7 * u.AA),
    )
