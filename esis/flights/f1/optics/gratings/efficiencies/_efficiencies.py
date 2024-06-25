import pathlib
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na

__all__ = [
    "efficiency_vs_wavelength",
    "efficiency_vs_x",
    "efficiency_vs_y",
    "efficiency_vs_angle_0deg",
    "efficiency_vs_angle_3deg",
]

_directory_data = pathlib.Path(__file__).parent / "_data"

serial_number = "UBO-16-017"
time_measurement = astropy.time.Time("2018-01-21")


def efficiency_vs_wavelength() -> (
    na.FunctionArray[na.TemporalSpectralDirectionalVectorArray, na.ScalarArray]
):
    """
    The total (coating + groove) efficiency of the ESIS diffraction gratings
    as a function of wavelength as measured by Eric Gullikson.

    Examples
    --------
    Plot the efficiency vs wavelength measurements using matplotlib.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        from esis.flights.f1.optics import gratings

        # Load the efficiency measurements
        efficiency = gratings.efficiencies.efficiency_vs_wavelength()

        # Plot the measurements using matplotlib
        fig, ax = plt.subplots()
        na.plt.plot(
            efficiency.inputs.wavelength,
            efficiency.outputs,
            ax=ax,
            label=efficiency.inputs.time.strftime("%Y-%m-%d"),
        );
        ax.set_xlabel(f"wavelength ({efficiency.inputs.wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    wavelength, efficiency = np.loadtxt(
        fname=_directory_data / "mul063315.abs",
        unpack=True,
        skiprows=1,
    )

    wavelength = (wavelength * u.nm).to(u.AA)

    wavelength = na.ScalarArray(wavelength, axes="wavelength")
    efficiency = na.ScalarArray(efficiency, axes="wavelength")

    return na.FunctionArray(
        inputs=na.TemporalSpectralDirectionalVectorArray(
            time=time_measurement,
            wavelength=wavelength,
            direction=0 * u.deg,
        ),
        outputs=efficiency,
    )


def efficiency_vs_x() -> (
    na.FunctionArray[na.TemporalSpectralPositionalVectorArray, na.ScalarArray]
):
    """
    The total (coating + groove) efficiency of the ESIS diffraction gratings
    as a function of :math:`x` position as measured by Eric Gullikson.

    Examples
    --------
    Plot the efficiency vs :math:`x` position measurements using matplotlib.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        from esis.flights.f1.optics import gratings

        # Load the efficiency measurements
        efficiency = gratings.efficiencies.efficiency_vs_x()

        # Plot the measurements using matplotlib
        fig, ax = plt.subplots()
        na.plt.plot(
            efficiency.inputs.position,
            efficiency.outputs,
            ax=ax,
            label=efficiency.inputs.time.strftime("%Y-%m-%d"),
        );
        ax.set_xlabel(f"$x$ ({efficiency.inputs.position.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    x, efficiency = np.loadtxt(
        fname=_directory_data / "mul063283.abs",
        unpack=True,
        skiprows=1,
    )

    x = x << u.mm

    x = na.ScalarArray(x, axes="grating_x")
    efficiency = na.ScalarArray(efficiency, axes="grating_x")

    return na.FunctionArray(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=time_measurement,
            wavelength=63 * u.nm,
            position=x,
        ),
        outputs=efficiency,
    )


def efficiency_vs_y() -> (
    na.FunctionArray[na.TemporalSpectralPositionalVectorArray, na.ScalarArray]
):
    """
    The total (coating + groove) efficiency of the ESIS diffraction gratings
    as a function of :math:`y` position as measured by Eric Gullikson.

    Examples
    --------
    Plot the efficiency vs :math:`y` position measurements using matplotlib.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        from esis.flights.f1.optics import gratings

        # Load the efficiency measurements
        efficiency = gratings.efficiencies.efficiency_vs_y()

        # Plot the measurements using matplotlib
        fig, ax = plt.subplots()
        na.plt.plot(
            efficiency.inputs.position,
            efficiency.outputs,
            ax=ax,
            label=efficiency.inputs.time.strftime("%Y-%m-%d"),
        );
        ax.set_xlabel(f"$y$ ({efficiency.inputs.position.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    y, efficiency = np.loadtxt(
        fname=_directory_data / "mul063282.txt",
        unpack=True,
        skiprows=1,
    )

    y = y << u.mm

    y = na.ScalarArray(y, axes="grating_y")
    efficiency = na.ScalarArray(efficiency, axes="grating_y")

    return na.FunctionArray(
        inputs=na.TemporalSpectralPositionalVectorArray(
            time=time_measurement,
            wavelength=63 * u.nm,
            position=y,
        ),
        outputs=efficiency,
    )


def efficiency_vs_angle_0deg() -> (
    na.FunctionArray[na.TemporalSpectralDirectionalVectorArray, na.ScalarArray]
):
    """
    The total (coating + groove) efficiency of the ESIS diffraction gratings
    as a function of output angle as measured by Eric Gullikson at an input
    angle of 0 degrees.

    Examples
    --------
    Plot the efficiency vs output angle measurements using matplotlib.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        from esis.flights.f1.optics import gratings

        # Load the efficiency measurements
        efficiency = gratings.efficiencies.efficiency_vs_angle_0deg()

        # Plot the measurements using matplotlib
        fig, ax = plt.subplots()
        na.plt.plot(
            efficiency.inputs.direction.output,
            efficiency.outputs,
            ax=ax,
            label=efficiency.inputs.direction.input,
        );
        ax.set_xlabel(f"output angle ({efficiency.inputs.direction.input.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    angle, efficiency = np.loadtxt(
        fname=_directory_data / "mul063284.txt",
        unpack=True,
        skiprows=1,
    )

    angle = angle << u.deg

    angle = na.ScalarArray(angle, axes="grating_output_angle")
    efficiency = na.ScalarArray(efficiency, axes="grating_output_angle")

    return na.FunctionArray(
        inputs=na.TemporalSpectralDirectionalVectorArray(
            time=time_measurement,
            wavelength=63 * u.nm,
            direction=na.InputOutputVectorArray(
                input=0 * u.deg,
                output=angle,
            ),
        ),
        outputs=efficiency,
    )


def efficiency_vs_angle_3deg() -> (
    na.FunctionArray[na.TemporalSpectralDirectionalVectorArray, na.ScalarArray]
):
    """
    The total (coating + groove) efficiency of the ESIS diffraction gratings
    as a function of output angle as measured by Eric Gullikson at an input
    angle of 3 degrees.

    Examples
    --------
    Plot the efficiency vs output angle measurements using matplotlib.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        from esis.flights.f1.optics import gratings

        # Load the efficiency measurements
        efficiency = gratings.efficiencies.efficiency_vs_angle_3deg()

        # Plot the measurements using matplotlib
        fig, ax = plt.subplots()
        na.plt.plot(
            efficiency.inputs.direction.output,
            efficiency.outputs,
            ax=ax,
            label=efficiency.inputs.direction.input,
        );
        ax.set_xlabel(f"output angle ({efficiency.inputs.direction.input.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    angle, efficiency = np.loadtxt(
        fname=_directory_data / "mul063281.txt",
        unpack=True,
        skiprows=1,
    )

    angle = angle << u.deg

    angle = na.ScalarArray(angle, axes="grating_output_angle")
    efficiency = na.ScalarArray(efficiency, axes="grating_output_angle")

    return na.FunctionArray(
        inputs=na.TemporalSpectralDirectionalVectorArray(
            time=time_measurement,
            wavelength=63 * u.nm,
            direction=na.InputOutputVectorArray(
                input=3 * u.deg,
                output=angle,
            ),
        ),
        outputs=efficiency,
    )
