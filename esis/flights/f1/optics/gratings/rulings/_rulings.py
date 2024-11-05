import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from esis.flights.f1.optics import gratings

__all__ = [
    "ruling_design",
    "ruling_measurement",
]


def ruling_design(
    num_distribution: int = 11,
) -> optika.rulings.AbstractRulings:
    """
    The as-designed rulings for the ESIS diffraction gratings.

    Parameters
    ----------
    num_distribution
        The number of Monte Carlo samples to draw when computing uncertainties.

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
            num_distribution=num_distribution,
        ),
        ratio_duty=na.UniformUncertainScalarArray(
            nominal=0.5,
            width=0.1,
            num_distribution=num_distribution,
        ),
        diffraction_order=-1,
    )


def ruling_measurement(
    num_distribution: int = 11,
):
    """
    A model of the rulings where the efficiency has been calculated using the
    ratio of the total efficiency of the gratings to the efficiency of the
    multilayer coatings.

    The total efficiency of the gratings is given by
    :func:`esis.flights.f1.optics.gratings.efficiencies.efficiency_vs_wavelength()`,
    and the efficiency of the multilayer coating is given by
    :func:`esis.flights.f1.optics.gratings.materials.multilayer_fit()`.

    Parameters
    ----------
    num_distribution
        The number of Monte Carlo samples to draw when computing uncertainties.

    Examples
    --------

    Compare the as-designed efficiency of the rulings to the measured efficiency
    of the rulings.

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

        # Define the surface normal
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        # Initialize the ESIS diffraction grating ruling model
        ruling_design = gratings.rulings.ruling_design(num_distribution=0)
        ruling_measurement = gratings.rulings.ruling_measurement(num_distribution=0)

        # Compute the efficiency of the grating rulings
        efficiency_design = ruling_design.efficiency(
            rays=rays,
            normal=normal,
        )
        efficiency_measurement = ruling_measurement.efficiency(
            rays=rays,
            normal=normal,
        )

        # Plot the efficiency vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            efficiency_design,
            ax=ax,
            color="tab:blue",
            label="design",
        );
        na.plt.plot(
            wavelength,
            efficiency_measurement,
            ax=ax,
            color="tab:orange",
            label="measurement",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("efficiency");
        ax.legend();
    """

    design = ruling_design(num_distribution=num_distribution)

    density = na.UniformUncertainScalarArray(
        nominal=2585.5 / u.mm,
        width=1 / u.mm,
        num_distribution=num_distribution,
    )

    spacing = design.spacing
    spacing.coefficients[0] = 1 / density

    efficiency_total = gratings.efficiencies.efficiency_vs_wavelength()

    wavelength = efficiency_total.inputs.wavelength
    angle = efficiency_total.inputs.direction

    coating = gratings.materials.multilayer_fit()

    efficiency_coating = coating.efficiency(
        rays=optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        ),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )

    efficiency_rulings = na.FunctionArray(
        inputs=efficiency_total.inputs,
        outputs=efficiency_total.outputs / efficiency_coating,
    )

    return optika.rulings.MeasuredRulings(
        spacing=spacing,
        diffraction_order=design.diffraction_order,
        efficiency_measured=efficiency_rulings,
    )
