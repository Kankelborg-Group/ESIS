import typing as typ
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u
from kgpy import Name, optics
from .. import Component

__all__ = ['Grating']

AperSurfT = optics.surface.Toroidal[None, optics.aperture.IsoscelesTrapezoid]
MainSurfT = optics.surface.ToroidalVariableLineSpaceGrating[optics.material.Mirror, optics.aperture.IsoscelesTrapezoid]

default_name = Name('grating')


@dataclasses.dataclass
class Grating(Component):
    name: Name = dataclasses.field(default_factory=lambda: default_name)
    tangential_radius: u.Quantity = np.inf * u.mm
    sagittal_radius: u.Quantity = np.inf * u.mm
    nominal_input_angle: u.Quantity = 0 * u.deg
    nominal_output_angle: u.Quantity = 0 * u.deg
    groove_density: u.Quantity = 0 / u.mm
    groove_density_coeff_linear: u.Quantity = 0 / (u.mm ** 2)
    groove_density_coeff_quadratic: u.Quantity = 0 / (u.mm ** 3)
    groove_density_coeff_cubic: u.Quantity = 0 / (u.mm ** 4)
    piston: u.Quantity = 0 * u.mm
    channel_radius: u.Quantity = 0 * u.mm
    channel_angle: u.Quantity = 0 * u.deg
    inclination: u.Quantity = 0 * u.deg
    aper_half_angle: u.Quantity = 0 * u.deg
    aper_decenter_x: u.Quantity = 0 * u.mm
    inner_clear_radius: u.Quantity = 0 * u.mm
    outer_clear_radius: u.Quantity = 0 * u.mm
    inner_border_width: u.Quantity = 0 * u.mm
    outer_border_width: u.Quantity = 0 * u.mm
    side_border_width: u.Quantity = 0 * u.mm
    dynamic_clearance: u.Quantity = 0 * u.mm
    substrate_thickness: u.Quantity = 0 * u.mm

    @classmethod
    def from_gregorian_layout(
            cls,
            magnification: float,
            primary_focal_length: u.Quantity,
            primary_clear_radius: u.Quantity,
            detector_channel_radius: u.Quantity,
            detector_piston: u.Quantity,
            grating_mechanical_margin: u.Quantity,
    ) -> 'Grating':
        """
        Computes an optimal placement for the grating based on the magnification, the primary location and the detector
        location.
        Based off of the procedure developed by Charles Kankelborg in SPIDER/spider/optics/design/configurations.ipynb
        :param magnification: Requested magnification of the Gregorian telescope
        :param primary_focal_length: Focal length of the parabolic primary mirror.
        :param primary_clear_radius: Radius of the usable area on the primary mirror
        :param detector_channel_radius: Radial distance from the center of the detector to the axis of symmetry
        :param detector_piston: Distance between the apex of the primary and the center of the detector along the axis
        of symmetry
        :param grating_mechanical_margin: Size of the unusable border around the outside of the grating.
        :return: A new grating instance with the placement consistent with the specified Gregorian system.
        """
        M = magnification
        f = primary_focal_length.to(u.mm).value
        D_p = 2 * primary_clear_radius.to(u.mm).value
        r_d = detector_channel_radius.to(u.mm).value
        x_d = detector_piston.to(u.mm).value
        m_g = grating_mechanical_margin.to(u.mm).value

        def gregorian_system(prams: typ.Tuple[float, ...]) -> typ.Tuple[float, ...]:
            x_g, r_g, D_g, h_g = prams
            eq1 = D_p * (x_g - f) - D_g * f
            eq2 = (D_p - D_g - 2 * m_g) * (x_g - f) - 2 * f * h_g
            eq3 = 2 * r_g - D_g + h_g
            eq4 = M ** 2 * ((x_g - f) ** 2 + r_g ** 2) - ((x_g - x_d) ** 2 + (r_d - r_g) ** 2)
            return eq1, eq2, eq3, eq4

        x_g = 1.2 * f
        r_g = D_p / 6
        D_g = D_p * (x_g - f) / f
        h_g = D_g / 4
        x_g, r_g, D_g, h_g = scipy.optimize.fsolve(gregorian_system, (x_g, r_g, D_g, h_g))
        x_g, r_g, D_g, h_g = x_g << u.mm, r_g << u.mm, D_g << u.mm, h_g << u.mm

        return cls(
            piston=x_g,
            channel_radius=r_g,
            aper_decenter_x=-r_g,
            inner_clear_radius=D_g / 2 - h_g,
            outer_clear_radius=D_g / 2,
        )

    @property
    def dynamic_clearance_x(self):
        return self.dynamic_clearance / np.sin(self.aper_half_angle)

    @property
    def surface(self) -> AperSurfT:
        side_border_x = self.side_border_width / np.sin(self.aper_half_angle) + self.dynamic_clearance_x
        return optics.surface.Toroidal(
            name=Name('aper'),
            radius=self.sagittal_radius,
            aperture=optics.aperture.IsoscelesTrapezoid(
                decenter=optics.coordinate.Decenter(x=self.aper_decenter_x + side_border_x),
                inner_radius=self.inner_clear_radius - side_border_x,
                outer_radius=self.outer_clear_radius - side_border_x,
                wedge_half_angle=self.aper_half_angle,
            ),
            radius_of_rotation=self.tangential_radius,
        )

    @property
    def _surfaces(self) -> optics.surface.Transformed[optics.surface.Substrate[AperSurfT, MainSurfT]]:
        return optics.surface.Transformed(
            name=self.name,
            surfaces=optics.surface.Substrate(
                aperture_surface=self.surface,
                main_surface=optics.surface.ToroidalVariableLineSpaceGrating(
                    name=Name('main'),
                    radius=self.sagittal_radius,
                    material=optics.material.Mirror(thickness=-self.substrate_thickness),
                    aperture=optics.aperture.IsoscelesTrapezoid(
                        is_active=False,
                        decenter=optics.coordinate.Decenter(x=self.aper_decenter_x + self.dynamic_clearance_x),
                        inner_radius=self.inner_clear_radius - self.inner_border_width - self.dynamic_clearance_x,
                        outer_radius=self.outer_clear_radius + self.outer_border_width - self.dynamic_clearance_x,
                        wedge_half_angle=self.aper_half_angle,
                    ),
                    radius_of_rotation=self.tangential_radius,
                    diffraction_order=1 * u.dimensionless_unscaled,
                    groove_density=self.groove_density,
                    coeff_linear=self.groove_density_coeff_linear,
                    coeff_quadratic=self.groove_density_coeff_quadratic,
                    coeff_cubic=self.groove_density_coeff_cubic,
                )
            ),
            transforms=[
                optics.coordinate.Transform(
                    translate=optics.coordinate.Translate(z=-self.piston)
                ),
                optics.coordinate.Transform(
                    tilt=optics.coordinate.Tilt(z=self.channel_angle),
                    translate=optics.coordinate.Translate(x=self.channel_radius),
                    tilt_first=True,
                ),
                optics.coordinate.Transform(
                    tilt=optics.coordinate.Tilt(y=self.inclination)
                ),
            ],
        )

    def copy(self) -> 'Grating':
        return Grating(
            tangential_radius=self.tangential_radius.copy(),
            sagittal_radius=self.sagittal_radius.copy(),
            groove_density=self.groove_density.copy(),
            piston=self.piston.copy(),
            channel_radius=self.channel_radius.copy(),
            channel_angle=self.channel_angle.copy(),
            inclination=self.inclination.copy(),
            aper_half_angle=self.aper_half_angle.copy(),
            aper_decenter_x=self.aper_decenter_x.copy(),
            inner_clear_radius=self.inner_clear_radius.copy(),
            outer_clear_radius=self.outer_clear_radius.copy(),
            inner_border_width=self.inner_border_width.copy(),
            outer_border_width=self.outer_border_width.copy(),
            side_border_width=self.side_border_width.copy(),
            substrate_thickness=self.substrate_thickness.copy(),
            name=self.name.copy(),
            groove_density_coeff_linear=self.groove_density_coeff_linear.copy(),
            groove_density_coeff_quadratic=self.groove_density_coeff_quadratic.copy(),
            groove_density_coeff_cubic=self.groove_density_coeff_cubic.copy(),
        )
