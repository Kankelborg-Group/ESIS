import named_arrays as na
import astropy.units as u
import optika

__all__ = [
    "ruling_design",
]


def ruling_design(
    num_distribution: int = 11,
) -> optika.rulings.AbstractRulings:
    """
    The as-designed rulings for the ESIS diffraction gratings.

    Examples
    --------

    Plot the efficiency of the rulings over the EUV wavelength range.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        from esis.flights.f1.optics import gratings

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(500, 700, axis="wavelength", num=1001) * u.AA

        # Define the incidence angle to be the same as the Horiba technical proposal
        angle = 1.3 * u.deg

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            position=0 * u.mm,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Initialize the ESIS diffraction grating ruling model
        rulings = gratings.rulings.ruling_design()

        # Compute the efficiency of the grating rulings
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the efficiency vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, efficiency, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("efficiency");
    """

    density = na.UniformUncertainScalarArray(
        nominal=(2.586608603456000 / u.um).to(1 / u.mm),
        width=1 / u.mm,
        num_distribution=num_distribution,
    )

    return optika.rulings.RectangularRulings(
        spacing=optika.rulings.Polynomial1dRulingSpacing(
            coefficients={
                0: 1 / density,
                1: na.UniformUncertainScalarArray(
                    nominal=-3.3849e-5 * (u.um / u.mm),
                    width=0.0512e-5 * (u.um / u.mm),
                    num_distribution=num_distribution,
                ),
                2: na.UniformUncertainScalarArray(
                    nominal=-1.3625e-7 * (u.um / u.mm**2),
                    width=0.08558e-7 * (u.um / u.mm**2),
                    num_distribution=num_distribution,
                ),
            },
            normal=na.Cartesian3dVectorArray(1, 0, 0),
        ),
        depth=na.UniformUncertainScalarArray(
            nominal=15 * u.nm,
            width=2 * u.nm,
        ),
        ratio_duty=na.UniformUncertainScalarArray(
            nominal=0.5,
            width=0.1,
        ),
        diffraction_order=1,
    )
