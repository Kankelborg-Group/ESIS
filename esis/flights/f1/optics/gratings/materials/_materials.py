import dataclasses
import pathlib
import numpy as np
import scipy
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "multilayer_design",
    "multilayer_witness_measured",
    "multilayer_witness_fit",
    "multilayer_fit",
]


def multilayer_design() -> optika.materials.MultilayerMirror:
    """
    The as-designed multilayer coating for the ESIS diffraction gratings.
    This coating is based on the design outlined in :cite:t:`Soufli2012`.

    Based on the analysis of :cite:t:`Rebellato2018`, this model uses
    :cite:t:`Kortright1988` for the silicon carbide optical constants, and
    :cite:t:`VidalDasilva2010` for the magnesium optical constants.

    Examples
    --------

    Plot the reflectivity of the coating over the EUV wavelength range.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        from esis.flights.f1.optics import gratings

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Initialize the ESIS diffraction grating material
        material = gratings.materials.multilayer_design()

        # Compute the reflectivity of the grating multilayer coating
        reflectivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, reflectivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");

    Plot a diagram of the multilayer stack

    .. jupyter-execute::

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            material.plot_layers(
                ax=ax,
                thickness_substrate=20 * u.nm,
            )
            ax.set_axis_off()
    """

    layer_oxide = optika.materials.Layer(
        chemical="SiO2",
        thickness=1 * u.nm,
        interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
        kwargs_plot=dict(
            color="gray",
        ),
        x_label=1.1,
    )

    layer_sic = optika.materials.Layer(
        chemical=optika.chemicals.Chemical(
            formula="SiC",
            is_amorphous=True,
            table="kortright",
        ),
        thickness=10 * u.nm,
        interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
        kwargs_plot=dict(
            color="lightgray",
        ),
    )

    layer_al = optika.materials.Layer(
        chemical="Al",
        thickness=1 * u.nm,
        interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
        kwargs_plot=dict(
            color="lightblue",
        ),
        x_label=1.1,
    )

    layer_mg = optika.materials.Layer(
        chemical=optika.chemicals.Chemical(
            formula="Mg",
            table="fernandez_perea",
        ),
        thickness=30 * u.nm,
        interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
        kwargs_plot=dict(
            color="pink",
        ),
    )

    layer_substrate = optika.materials.Layer(
        chemical="SiO2",
        thickness=10 * u.mm,
        interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
        kwargs_plot=dict(
            color="gray",
        ),
    )

    return optika.materials.MultilayerMirror(
        layers=[
            layer_oxide,
            layer_sic,
            dataclasses.replace(
                layer_al,
                thickness=4 * u.nm,
                interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
            ),
            layer_mg,
            optika.materials.PeriodicLayerSequence(
                layers=[
                    layer_al,
                    layer_sic,
                    layer_mg,
                ],
                num_periods=3,
            ),
            dataclasses.replace(
                layer_al,
                thickness=10 * u.nm,
                interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
            ),
        ],
        substrate=layer_substrate,
    )


def multilayer_witness_measured() -> optika.materials.MeasuredMirror:
    """
    A reflectivity measurement of the witness samples associated with the ESIS
    diffraction gratings gathered by Eric Gullikson.

    Examples
    --------
    Load the measurement and plot it as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        from esis.flights.f1.optics import gratings

        # Load the witness sample measurements
        multilayer = gratings.materials.multilayer_witness_measured()
        measurement = multilayer.efficiency_measured

        # Plot the measurement as a function of wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            measurement.inputs.wavelength,
            measurement.outputs,
            ax=ax,
            axis="wavelength",
            label=multilayer.serial_number,
        )
        ax.set_xlabel(f"wavelength ({measurement.inputs.wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
        ax.legend();
    """
    path_base = pathlib.Path(__file__).parent / "_data"

    wavelength_17, reflectivity_17 = np.loadtxt(
        fname=path_base / "Witness_g17.txt",
        skiprows=1,
        unpack=True,
    )
    wavelength_19, reflectivity_19 = np.loadtxt(
        fname=path_base / "Witness_g19.txt",
        skiprows=1,
        unpack=True,
    )
    wavelength_24, reflectivity_24 = np.loadtxt(
        fname=path_base / "Witness_g24.txt",
        skiprows=1,
        unpack=True,
    )

    wavelength_17 = na.ScalarArray(wavelength_17, axes="wavelength")
    wavelength_19 = na.ScalarArray(wavelength_19, axes="wavelength")
    wavelength_24 = na.ScalarArray(wavelength_24, axes="wavelength")

    reflectivity_17 = na.ScalarArray(reflectivity_17, axes="wavelength")
    reflectivity_19 = na.ScalarArray(reflectivity_19, axes="wavelength")
    reflectivity_24 = na.ScalarArray(reflectivity_24, axes="wavelength")

    wavelength = na.stack(
        arrays=[wavelength_17, wavelength_19, wavelength_24],
        axis="channel",
    )
    reflectivity = na.stack(
        arrays=[reflectivity_17, reflectivity_19, reflectivity_24],
        axis="channel",
    )

    wavelength = (wavelength << u.nm) << u.AA

    serial_number = [
        "UBO-16-017",
        "UBO-16-019",
        "UBO-16-024",
    ]
    serial_number = np.array(serial_number)
    serial_number = na.ScalarArray(serial_number, axes="channel")

    result = optika.materials.MeasuredMirror(
        efficiency_measured=na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=4 * u.deg,
            ),
            outputs=reflectivity,
        ),
        substrate=optika.materials.Layer(
            chemical="Si",
        ),
        serial_number=serial_number,
    )

    return result


def multilayer_witness_fit() -> optika.materials.MultilayerMirror:
    r"""
    A multilayer stack fitted to the witness sample measurements given by
    :func:`multilayer_witness_measured`.

    This fit has five free parameters: the ratio of the thicknesses of
    :math:`\text{Mg}`, :math:`\text{Al}`, and the :math:`\text{SiC}` to their
    as-designed thickness, the roughness of the substrate, and a single
    roughness parameter for all the layers in the multilayer stack.

    Examples
    --------

    Plot the fitted vs. measured reflectivity of the grating witness samples.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        from esis.flights.f1.optics import gratings

        # Load the measured reflectivity of the witness samples
        multilayer_measured = gratings.materials.multilayer_witness_measured()
        measurement = multilayer_measured.efficiency_measured

        # Isolate the angle of incidence of the measurement
        angle_incidence = measurement.inputs.direction

        # Fit a multilayer stack to the measured reflectivity
        multilayer = gratings.materials.multilayer_witness_fit()

        # Define the rays incident on the multilayer stack that will be used to
        # compute the reflectivity
        rays = optika.rays.RayVectorArray(
            wavelength=na.geomspace(250, 950, axis="wavelength", num=1001) * u.AA,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle_incidence),
                y=0,
                z=np.cos(angle_incidence),
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
            measurement.inputs.wavelength,
            measurement.outputs,
            ax=ax,
            label="measurement"
        );
        na.plt.plot(
            rays.wavelength,
            reflectivity_fit,
            ax=ax,
            axis="wavelength",
            label="fit",
            color="tab:orange",
        );
        ax.set_xlabel(f"wavelength ({rays.wavelength.unit:latex_inline})")
        ax.set_ylabel("reflectivity")
        ax.legend();

        # Print the fitted multilayer stack
        multilayer

    Plot a diagram of the fitted multilayer stack

    .. jupyter-execute::

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            multilayer.plot_layers(
                ax=ax,
                thickness_substrate=20 * u.nm,
            )
            ax.set_axis_off()

    """

    design = multilayer_design()

    measurement = multilayer_witness_measured()
    unit = u.nm

    reflectivity = measurement.efficiency_measured.outputs
    angle_incidence = measurement.efficiency_measured.inputs.direction

    rays = optika.rays.RayVectorArray(
        wavelength=measurement.efficiency_measured.inputs.wavelength,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle_incidence),
            y=0,
            z=np.cos(angle_incidence),
        ),
    )

    normal = na.Cartesian3dVectorArray(0, 0, -1)

    def _multilayer(
        ratio_SiC: float,
        ratio_Al: float,
        ratio_Mg: float,
        width_layers: float,
        width_substrate: float,
    ):

        width_layers = width_layers * unit
        width_substrate = width_substrate * unit

        result = multilayer_design()

        result.layers[1].thickness = ratio_SiC * design.layers[1].thickness
        result.layers[1].interface.width = width_layers

        result.layers[2].thickness = ratio_Al * design.layers[2].thickness
        result.layers[2].interface.width = width_layers

        result.layers[3].thickness = ratio_Mg * design.layers[3].thickness
        result.layers[3].interface.width = width_layers

        thickness_Al = ratio_Al * design.layers[~1].layers[0].thickness
        result.layers[~1].layers[0].thickness = thickness_Al
        result.layers[~1].layers[0].interface.width = width_layers

        thickness_SiC = ratio_SiC * design.layers[~1].layers[~1].thickness
        result.layers[~1].layers[~1].thickness = thickness_SiC
        result.layers[~1].layers[~1].interface.width = width_layers

        thickness_Mg = ratio_Mg * design.layers[~1].layers[~0].thickness
        result.layers[~1].layers[~0].thickness = thickness_Mg
        result.layers[~1].layers[~0].interface.width = width_layers

        result.layers[~0].thickness = ratio_Al * design.layers[~0].thickness
        result.layers[~0].interface.width = width_layers

        result.layers.pop(0)

        result.substrate.interface.width = width_substrate
        result.substrate.chemical = "Si"

        return result

    def _func(x: np.ndarray):

        multilayer = _multilayer(*x)

        reflectivity_fit = multilayer.efficiency(
            rays=rays,
            normal=normal,
        )

        result = np.sqrt(np.nanmean(np.square(reflectivity_fit - reflectivity)))

        return result.ndarray.value

    x0 = u.Quantity(
        [
            1,
            1,
            1,
            1,
            1,
        ]
    )

    bounds = [(0, None)] * len(x0)

    fit = scipy.optimize.minimize(
        fun=_func,
        x0=x0,
        bounds=bounds,
    )

    return _multilayer(*fit.x)


def multilayer_fit() -> optika.materials.MultilayerMirror:
    """
    A multilayer coating determined by modifying :func:`multilayer_witness_fit`
    to have a glass substrate with the appropriate roughness.

    Examples
    --------

    Plot the theoretical reflectivity of this multilayer stack vs. the
    theoretical reflectivity of :func:`multilayer_witness_fit`.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        from esis.flights.f1.optics import gratings

        # Define a grid of wavelength samples
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define a grid of incidence angles
        angle = 4 * u.deg

        # Define the light rays incident on the multilayer stack
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Initialize the multilayer stacks
        multilayer_witness_fit = gratings.materials.multilayer_witness_fit()
        multilayer_fit = gratings.materials.multilayer_fit()

        # Define the vector normal to the multilayer stack
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        # Compute the reflectivity of the multilayer for the given incident rays
        reflectivity_witness = multilayer_witness_fit.efficiency(rays, normal)
        reflectivity_fit = multilayer_fit.efficiency(rays, normal)

        # Plot the reflectivities as a function of wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            reflectivity_witness,
            ax=ax,
            axis="wavelength",
            label="witness fit",
        );
        na.plt.plot(
            wavelength,
            reflectivity_fit,
            ax=ax,
            axis="wavelength",
            label="grating fit",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");
        ax.legend();

    |

    Plot the reflectivity of this multilayer stack as a function of incidence angle.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika
        from esis.flights.f1.optics import gratings

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
        multilayer = gratings.materials.multilayer_fit()

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
    result = multilayer_design()
    witness = multilayer_witness_fit()

    result.layers = result.layers[:1] + witness.layers

    result.substrate.interface.width = 0.5 * u.nm

    return result
