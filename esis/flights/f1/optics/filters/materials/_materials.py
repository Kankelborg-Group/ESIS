import astropy.units as u
import optika

__all__ = ["thin_film_design"]


def thin_film_design() -> optika.materials.ThinFilmFilter:
    """
    The as-designed thin-film visible light filter material for the ESIS instrument.

    Examples
    --------

    Plot the transmissivity of the thin-film filter over the EUV wavelength
    range.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika
        from esis.flights.f1.optics import filters

        # Define an array of wavelengths with which to sample the efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incident rays from the wavelength array
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Initialize the ESIS diffraction grating material
        material = filters.materials.thin_film_design()

        # Compute the reflectivity of the primary mirror
        transmissivity = material.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the transmissivity vs wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, transmissivity, ax=ax);
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("transmissivity");
    """

    return optika.materials.ThinFilmFilter(
        layer=optika.materials.Layer(
            chemical="Al",
            thickness=100 * u.nm,
        ),
        layer_oxide=optika.materials.Layer(
            chemical="Al2O3",
            thickness=4 * u.nm,
        ),
        mesh=optika.materials.meshes.Mesh(
            chemical="Ni",
            efficiency=0.82,
            pitch=70 / u.imperial.inch,
        ),
    )
