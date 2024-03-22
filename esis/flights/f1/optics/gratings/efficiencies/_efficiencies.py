import pathlib
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na

__all__ = [
    "efficiency_vs_wavelength",
]

_directory_data = pathlib.Path(__file__).parent / "_data"

serial_number = "UBO-16-017"
time_measurement = date_measurement = astropy.time.Time('2018-01-21')


def efficiency_vs_wavelength(
) -> na.FunctionArray[na.TemporalSpectralDirectionalVectorArray, na.ScalarArray]:
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
        fname=_directory_data / 'mul063315.abs',
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
            direction=0 * u.deg
        ),
        outputs=efficiency,
    )
