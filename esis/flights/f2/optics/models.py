import astropy.units as u
import optika
import esis


def design_proposed(
    grid: None | optika.vectors.ObjectVectorArray = None,
    num_distribution: int = 11,
) -> esis.optics.OpticsModel:
    """
    Final optical design for ESIS 2.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties

    Examples
    --------
    Plot the rays traveling through the optical system, as viewed from the front.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        import esis

        grid = optika.vectors.ObjectVectorArray(
            wavelength=na.linspace(-1, 1, axis="wavelength",  num=2) / 2,
            field=0.99 * na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=5,
            ),
        )

        model = esis.flights.f2.optics.models.design_proposed(
            grid=grid,
            num_distribution=0,
        )

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                figsize=(6, 6.5),
                constrained_layout=True
            )
            ax.set_aspect("equal")
            model.system.plot(
                components=("x", "y"),
                color="black",
                kwargs_rays=dict(
                    color=na.ScalarArray(np.array(["tab:orange", "tab:blue"]), axes="wavelength"),
                    label=model.system.grid_input.wavelength.astype(int),
                ),
            );
            handles, labels = ax.get_legend_handles_labels()
            labels = dict(zip(labels, handles))
            fig.legend(labels.values(), labels.keys());
    """
    result = esis.flights.f1.optics.models.design_full(
        grid=grid,
        num_distribution=num_distribution,
    )

    c0 = 1 / (2700 / u.mm)
    c1 = -2.852e-5 * (u.um / u.mm)
    c2 = -2.112e-7 * (u.um / u.mm**2)

    result.grating.rulings.spacing.coefficients[0].nominal = c0
    result.grating.rulings.spacing.coefficients[1].nominal = c1
    result.grating.rulings.spacing.coefficients[2].nominal = c2

    result.grating.yaw = -3.65 * u.deg

    z_filter = result.grating.translation.z.nominal + 1291.012 * u.mm
    dz = z_filter - result.filter.translation.z
    result.filter.translation.z += dz
    result.detector.translation.z += dz

    return result

def design_proposed_single(
    grid: None | optika.vectors.ObjectVectorArray = None,
    num_distribution: int = 11,
) -> esis.optics.OpticsModel:
    """
    A single channel of the final optical design for ESIS 2.

    Parameters
    ----------
    grid
        sampling of wavelength, field, and pupil positions that will be used to
        characterize the optical system.
    num_distribution
        number of Monte Carlo samples to draw when computing uncertainties

    Examples
    --------
    Plot the rays traveling through the optical system, as viewed from the front.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        import esis

        grid = optika.vectors.ObjectVectorArray(
            wavelength=na.linspace(-1, 1, axis="wavelength",  num=2) / 2,
            field=0.99 * na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=5,
            ),
        )

        model = esis.flights.f2.optics.models.design_proposed(
            grid=grid,
            num_distribution=0,
        )

        fig, ax = plt.subplots(
            figsize=(6, 6.5),
            constrained_layout=True
        )
        ax.set_aspect("equal")
        model.system.plot(
            components=("x", "y"),
            color="black",
            kwargs_rays=dict(
                color=na.ScalarArray(np.array(["tab:orange", "tab:blue"]), axes="wavelength"),
                label=model.system.grid_input.wavelength.astype(int),
            ),
        );
        handles, labels = ax.get_legend_handles_labels()
        labels = dict(zip(labels, handles))
        fig.legend(labels.values(), labels.keys());
    """
    result = esis.flights.f1.optics.models.design_single(
        grid=grid,
        num_distribution=num_distribution,
    )

    c0 = 1 / (2700 / u.mm)
    c1 = -2.852e-5 * (u.um / u.mm)
    c2 = -2.112e-7 * (u.um / u.mm**2)

    result.grating.rulings.spacing.coefficients[0].nominal = c0
    result.grating.rulings.spacing.coefficients[1].nominal = c1
    result.grating.rulings.spacing.coefficients[2].nominal = c2

    result.grating.yaw = -3.65 * u.deg

    # commenting out to try and optimize around nominal camera position

    # z_filter = result.grating.translation.z.nominal + 1291.012 * u.mm
    # dz = z_filter - result.filter.translation.z
    # result.filter.translation.z += dz
    # result.detector.translation.z += dz

    return result
