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
    d = 257 * u.AA
    gamma = 0.5
    return optika.materials.MultilayerMirror(
        layers=[
            optika.materials.Layer(
                chemical="Al2O3",
                thickness=1 * u.nm,
                interface=optika.materials.profiles.ErfInterfaceProfile(
                    width=7 * u.AA,
                ),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.PeriodicLayerSequence(
                [
                    optika.materials.Layer(
                        chemical="Al",
                        thickness=gamma * d,
                        interface=optika.materials.profiles.ErfInterfaceProfile(
                            width=7 * u.AA,
                        ),
                        kwargs_plot=dict(
                            color="tab:blue",
                            alpha=0.5,
                        ),
                    ),
                    optika.materials.Layer(
                        chemical="Sc",
                        thickness=(1 - gamma) * d,
                        interface=optika.materials.profiles.ErfInterfaceProfile(
                            width=7 * u.AA,
                        ),
                        kwargs_plot=dict(
                            color="tab:orange",
                            alpha=0.5,
                        ),
                    ),
                ],
                num_periods=10,
            ),
        ],
        substrate=optika.materials.Layer(
            chemical="SiO2",
            thickness=30 * u.mm,
            interface=optika.materials.profiles.ErfInterfaceProfile(
                7 * u.AA,
            ),
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
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
    d = 260 * u.AA
    gamma = 0.5
    return optika.materials.MultilayerMirror(
        layers=[
            optika.materials.Layer(
                chemical="SiO2",
                thickness=1 * u.nm,
                interface=optika.materials.profiles.ErfInterfaceProfile(
                    width=7 * u.AA,
                ),
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.3,
                ),
            ),
            optika.materials.PeriodicLayerSequence(
                [
                    optika.materials.Layer(
                        chemical="Si",
                        thickness=gamma * d,
                        interface=optika.materials.profiles.ErfInterfaceProfile(
                            width=7 * u.AA,
                        ),
                        kwargs_plot=dict(
                            color="tab:blue",
                            alpha=0.5,
                        ),
                    ),
                    optika.materials.Layer(
                        chemical="Sc",
                        thickness=(1 - gamma) * d,
                        interface=optika.materials.profiles.ErfInterfaceProfile(
                            width=7 * u.AA,
                        ),
                        kwargs_plot=dict(
                            color="tab:orange",
                            alpha=0.5,
                        ),
                    ),
                ],
                num_periods=10,
            ),
        ],
        substrate=optika.materials.Layer(
            chemical="SiO2",
            thickness=30 * u.mm,
            interface=optika.materials.profiles.ErfInterfaceProfile(
                7 * u.AA,
            ),
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        ),
    )
