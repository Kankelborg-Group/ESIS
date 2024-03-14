import dataclasses
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "multilayer_design",
    "multilayer_witness_measured",
]


def multilayer_design() -> optika.materials.MultilayerMirror:
    """
    The as-designed multilayer coating for the ESIS diffraction gratings.

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

        # Compute the reflectivity of the primary mirror
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

    layer_sic = optika.materials.Layer(
        chemical="SiC",
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
        chemical="Mg",
        thickness=30 * u.nm,
        interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
        kwargs_plot=dict(
            color="pink",
        ),
    )

    layer_sio2 = optika.materials.Layer(
        chemical="SiO2",
        thickness=10 * u.mm,
        interface=optika.materials.profiles.ErfInterfaceProfile(1 * u.nm),
        kwargs_plot=dict(
            color="gray",
        ),
    )

    return optika.materials.MultilayerMirror(
        layers=[
            dataclasses.replace(
                layer_sio2,
                thickness=1 * u.nm,
                x_label=1.1,
            ),
            layer_sic,
            dataclasses.replace(layer_al, thickness=4 * u.nm),
            layer_mg,
            optika.materials.PeriodicLayerSequence(
                layers=[
                    layer_al,
                    layer_sic,
                    layer_mg,
                ],
                num_periods=3,
            ),
            dataclasses.replace(layer_al, thickness=10 * u.nm),
        ],
        substrate=layer_sio2,
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
