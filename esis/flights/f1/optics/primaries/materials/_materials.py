import pathlib
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "multilayer_design",
    "reflectivity_witness",
    "multilayer_witness",
    "multilayer_fit",
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
        material = esis.flights.f1.optics.primaries.materials.multilayer_design()

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


def reflectivity_witness() -> (
    na.FunctionArray[na.SpectralDirectionalVectorArray, na.ScalarArray]
):
    """
    A reflectivity measurement of the witness samples to the primary mirror
    multilayer coating performed by Eric Gullikson.

    Examples
    --------
    Load the measurement and plot it as a function of wavelength

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        import esis

        # Load the witness sample measurements
        measurement = esis.flights.f1.optics.primaries.materials.reflectivity_witness()

        # Plot the measurement as a function of wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            measurement.inputs.wavelength,
            measurement.outputs,
            ax=ax,
            label=measurement.inputs.direction,
        )
        ax.set_xlabel(f"wavelength ({measurement.inputs.wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
        ax.legend();
    """
    wavelength, reflectivity = np.loadtxt(
        fname=pathlib.Path(__file__).parent / "_data/mul063931.abs",
        skiprows=1,
        unpack=True,
    )
    wavelength = na.ScalarArray(wavelength << u.nm, axes="wavelength").to(u.AA)
    reflectivity = na.ScalarArray(reflectivity, axes="wavelength")
    result = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=4 * u.deg,
        ),
        outputs=reflectivity,
    )
    return result


def multilayer_witness() -> optika.materials.MultilayerMirror:
    """
    A multilayer stack fitted to the :func:`reflectivity_witness` measurement.

    Examples
    --------
    Plot the fitted vs. measured reflectivity of the primary mirror witness sample.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na
        import optika
        import esis

        # Load the measured reflectivity of the witness sample
        reflectivity_measured = esis.flights.f1.optics.primaries.materials.reflectivity_witness()

        # Fit a multilayer stack to the measured reflectivity
        multilayer = esis.flights.f1.optics.primaries.materials.multilayer_witness()

        # Define the rays incident on the multilayer stack that will be used to
        # compute the reflectivity
        rays = optika.rays.RayVectorArray(
            wavelength=reflectivity_measured.inputs.wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(reflectivity_measured.inputs.direction),
                y=0,
                z=np.cos(reflectivity_measured.inputs.direction),
            ),
        )

        # Compute the reflectivity of the fitted multilayer stack
        reflectivity_fit = multilayer.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the fitted vs. measured reflectivity
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.scatter(
            reflectivity_measured.inputs.wavelength,
            reflectivity_measured.outputs,
            ax=ax,
            label="measured"
        );
        na.plt.plot(
            rays.wavelength,
            reflectivity_fit,
            ax=ax,
            label="fitted",
            color="tab:orange",
        );
        ax.set_xlabel(f"wavelength ({rays.wavelength.unit:latex_inline})")
        ax.set_ylabel("reflectivity")
        ax.legend();

        # Print the fitted multilayer stack
        multilayer
    """

    design = multilayer_design()

    (
        guess_SiO2,
        guess_SiC,
        guess_Cr,
    ) = design.thickness_layers.ndarray

    guess_interface = 0.5 * u.nm

    type_profile = type(design.profile_interface)

    reflectivity = reflectivity_witness()
    unit = u.nm

    rays = optika.rays.RayVectorArray(
        wavelength=reflectivity.inputs.wavelength,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(reflectivity.inputs.direction),
            y=0,
            z=np.cos(reflectivity.inputs.direction),
        ),
    )

    normal = na.Cartesian3dVectorArray(0, 0, -1)

    def _multilayer(
        thickness_SiO2: float,
        thickness_SiC: float,
        thickness_Cr: float,
        thickness_interface: float,
    ):
        return optika.materials.MultilayerMirror(
            material_layers=design.material_layers,
            material_substrate="Si",
            thickness_layers=na.ScalarArray(
                ndarray=[
                    thickness_SiO2,
                    thickness_SiC,
                    thickness_Cr,
                ]
                * unit,
                axes=design.axis_layers,
            ),
            axis_layers=design.axis_layers,
            profile_interface=type_profile(thickness_interface * unit),
        )

    def _func(x: np.ndarray):

        multilayer = _multilayer(*x)

        reflectivity_fit = multilayer.efficiency(
            rays=rays,
            normal=normal,
        )

        result = np.sqrt(np.mean(np.square(reflectivity_fit - reflectivity.outputs)))

        return result.ndarray.value

    fit = scipy.optimize.minimize(
        fun=_func,
        x0=[
            guess_SiO2.to_value(unit),
            guess_SiC.to_value(unit),
            guess_Cr.to_value(unit),
            guess_interface.to_value(unit),
        ],
        bounds=[
            (0, None),
            (0, None),
            (guess_Cr.to_value(unit), guess_Cr.to_value(unit)),
            (0, None),
        ],
        method="nelder-mead",
    )

    return _multilayer(*fit.x)


def multilayer_fit() -> optika.materials.MultilayerMirror:
    """
    A multilayer coating determined by modifying :func:`multilayer_witness`
    to have a glass substrate.

    Examples
    --------

    Plot the reflectivity of this multilayer stack as a function of incidence angle.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        import esis

        # Define a grid of wavelength samples
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define a grid of incidence angles
        angle = na.linspace(0, 20, axis="angle", num=3) * u.deg

        # Define the light rays incident on the multilayer stack
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Initialize the multilayer stack
        multilayer = esis.flights.f1.optics.primaries.materials.multilayer_fit()

        # Compute the reflectivity of the multilayer for the given incident rays
        reflectivity = multilayer.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity of the multilayer as a function of wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            reflectivity,
            ax=ax,
            axis="wavelength",
            label=angle,
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
        ax.legend();
    """
    design = multilayer_design()
    result = multilayer_witness()
    result.material_substrate = design.material_substrate
    return result

