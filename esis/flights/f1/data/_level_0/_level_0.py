import msfc_ccd
import esis
from ... import nsroc
from .. import path_fits

__all__ = [
    "level_0",
]


def level_0(
    axis_time: str = "time",
    axis_channel: str = "channel",
    axis_x: str = "detector_x",
    axis_y: str = "detector_y",
) -> esis.data.Level_0:
    """
    All the raw images captured by ESIS during the 2019 flight.

    Parameters
    ----------
    axis_time
        The name of the logical axis representing time.
    axis_channel
        The name of the logical axis representing the different ESIS channels.
    axis_x
        The name of the logical axis representing the detector's long axis.
    axis_y
        The name of the logical axis representing the detector's short axis.

    Examples
    --------

    Load the Level-0 dataset into a :class:`esis.data.Level_0` instance.

    .. jupyter-execute::

        import IPython.display
        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import esis

        # Define the names of the logical axes
        # to use for constructing the Level-0 dataset
        axis_time = "time"
        axis_channel = "channel"

        # Load the Level-0 dataset into memory
        level_0 = esis.flights.f1.data.level_0(
            axis_time=axis_time,
            axis_channel=axis_channel,
        )

    Make a movie of three frames of the Level-0 dataset.

    .. jupyter-execute::

        # Define a slice of three frames near apogee
        index = {axis_time: slice(20, 23)}

        # Create a figure
        fig, axs = na.plt.subplots(
            axis_rows="rows",
            axis_cols="cols",
            nrows=level_0.shape[axis_channel] // 2,
            ncols=2,
            sharex=True,
            sharey=True,
            constrained_layout=True,
            figsize=(10, 5),
        )

        # Reorganize the axes into a flat array
        ax = axs.combine_axes(("rows", "cols"), axis_channel)
        ax = ax[{axis_channel: slice(None, None, -1)}]

        # Define the colormap
        colorizer = plt.Colorizer(
            norm=plt.Normalize(
                vmin=level_0.outputs.percentile(1).ndarray,
                vmax=level_0.outputs.percentile(99).ndarray,
            ),
        )

        # Animate the Level-0 dataset frames
        ani = na.plt.pcolormovie(
            level_0.inputs.time[index].mean(axis_channel),
            level_0.inputs.pixel.x,
            level_0.inputs.pixel.y,
            C=level_0.outputs[index],
            axis_time=axis_time,
            ax=ax,
            kwargs_pcolormesh=dict(
                colorizer=colorizer,
            ),
        )

        # Create labels for each axis
        na.plt.text(
            x=0.5,
            y=1.01,
            s=level_0.channel,
            transform=na.plt.transAxes(ax),
            ax=ax,
            ha="center",
            va="bottom",
        )
        na.plt.set_aspect("equal", ax=ax)
        na.plt.set_xlabel("detector $x$ (pix)", ax=axs[dict(rows=0)])
        na.plt.set_ylabel("detector $y$ (pix)", ax=axs[dict(cols=0)])

        # Plot the colorbar using the colormap
        plt.colorbar(
            mappable=plt.cm.ScalarMappable(colorizer=colorizer),
            ax=ax.ndarray,
            label="signal (DN)"
        )

        # Render the movie as a javascript animation
        plt.close(fig)
        IPython.display.HTML(ani.to_jshtml())

    Plot the FPGA temperatures over the flight.

    .. jupyter-execute::

        # Convert the time array from ISO to a Python :class:`datetime.datetime`
        # instance
        time = level_0.inputs.time
        time = time.replace(ndarray=time.ndarray.datetime)

        # Plot the result as a line plot
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                time,
                level_0.inputs.temperature_fpga,
                axis=axis_time,
                ax=ax,
                label=level_0.channel,
            )
            ax.set_ylabel(f"FPGA temperature ({ax.get_ylabel()})")
            ax.legend()
    """

    path = path_fits(
        axis_time=axis_time,
        axis_channel=axis_channel,
    )

    return esis.data.Level_0.from_fits(
        path=path,
        sensor=msfc_ccd.TeledyneCCD230(),
        axis_x=axis_x,
        axis_y=axis_y,
        timeline=nsroc.timeline(),
    )
