import typing as typ
import pathlib
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
import astropy.visualization
# import astropy.modeling
import scipy.stats
import scipy.interpolate
import scipy.optimize
# import scipy.signal
import kgpy.vector
import kgpy.atmosphere
import kgpy.model
import kgpy.obs
import kgpy.nsroc
import kgpy.observatories
from . import nsroc
import esis

__all__ = ['Level_0']


@dataclasses.dataclass
class Level_0(kgpy.obs.Image):
    cam_sn: typ.Optional[np.ndarray] = None
    global_index: typ.Optional[np.ndarray] = None
    requested_exposure_time: typ.Optional[u.Quantity] = None
    run_mode: typ.Optional[np.ndarray] = None
    status: typ.Optional[np.ndarray] = None
    fpga_temp: typ.Optional[u.Quantity] = None
    fpga_vccint_voltage: typ.Optional[u.Quantity] = None
    fpga_vccaux_voltage: typ.Optional[u.Quantity] = None
    fpga_vccbram_voltage: typ.Optional[u.Quantity] = None
    adc_temp: typ.Optional[u.Quantity] = None
    # detector: typ.Optional[esis.optics.Detector] = None
    optics: typ.Optional[esis.optics.Optics] = None
    trajectory: typ.Optional[kgpy.nsroc.Trajectory] = None
    timeline: typ.Optional[nsroc.Timeline] = None
    # num_dark_safety_frames: int = 1
    num_ignored_bias_columns: int = 20
    num_invalid_exposures: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.update()

    def update(self) -> typ.NoReturn:
        self._intensity_derivative = None
        self._intensity_nobias = None
        self._darks_nobias = None
        self._trajectory = None
        self._offset_optimized = None
        self._intensity_electrons_prelim = None
        self._stray_light_avg = None

    @classmethod
    def from_directory(
            cls,
            directory: pathlib.Path,
            # detector: esis.optics.Detector,
            optics: typ.Optional[esis.optics.Optics] = None,
            trajectory: typ.Optional[kgpy.nsroc.Trajectory] = None,
            timeline: typ.Optional[kgpy.nsroc.Timeline] = None,
            # num_dark_safety_frames: int = 1,
    ) -> 'Level_0':
        fits_list = np.array(list(directory.glob('*.fit*')))
        fits_list.sort()

        chan_str_ind = 4
        channels = np.array([int(f.name[chan_str_ind]) for f in fits_list])
        channels = np.unique(channels)
        num_channels = len(channels)

        fits_list = fits_list.reshape((num_channels, -1))
        fits_list = fits_list.transpose()
        num_exposures = len(fits_list)

        hdu = astropy.io.fits.open(fits_list[0, 0])[0]
        self = cls.zeros((num_exposures, num_channels) + hdu.data.shape)
        # self.detector = detector
        self.optics = optics
        self.trajectory = trajectory
        self.timeline = timeline
        # self.num_dark_safety_frames = num_dark_safety_frames

        for i in range(num_exposures):
            self.intensity_uncertainty[i] = optics.detector.readout_noise_image(num_channels)
            for c in range(num_channels):
                hdu = astropy.io.fits.open(fits_list[i, c])[0]
                self.intensity[i, c] = hdu.data * u.adu
                self.time[i, c] = astropy.time.Time(hdu.header['IMG_TS'])
                self.exposure_length[i, c] = float(hdu.header['MEAS_EXP']) * 25 * u.ns
                if i == 0:
                    self.channel[c] = int(hdu.header['CAM_ID'][~0]) * u.chan
                if c == 0:
                    self.time_index[i] = int(hdu.header['IMG_CNT'])
                    # self.time_index[i] = int(hdu.header['IMG_CNT']) - self.num_invalid_exposures
                self.cam_sn[i, c] = int(hdu.header['CAM_SN'])
                self.global_index[i, c] = int(hdu.header['IMG_ISN'])
                self.requested_exposure_time[i, c] = float(hdu.header['IMG_EXP']) * u.ms
                self.run_mode[i, c] = hdu.header['RUN_MODE']
                self.status[i, c] = hdu.header['IMG_STAT']
                self.fpga_temp[i, c] = float(hdu.header['FPGATEMP']) * u.adu
                self.fpga_vccint_voltage[i, c] = float(hdu.header['FPGAVINT']) * u.adu
                self.fpga_vccaux_voltage[i, c] = float(hdu.header['FPGAVAUX']) * u.adu
                self.fpga_vccbram_voltage[i, c] = float(hdu.header['FPGAVBRM']) * u.adu
                self.adc_temp[i, c, 0] = float(hdu.header['ADCTEMP1']) * u.adu
                self.adc_temp[i, c, 1] = float(hdu.header['ADCTEMP2']) * u.adu
                self.adc_temp[i, c, 2] = float(hdu.header['ADCTEMP3']) * u.adu
                self.adc_temp[i, c, 3] = float(hdu.header['ADCTEMP4']) * u.adu

        self.time = self.time - self.exposure_half_length

        return self

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]) -> 'Level_0':
        sh = shape[:2]
        self = super().zeros(shape)  # type: Level_0
        self.cam_sn = np.zeros(sh, dtype=int)
        self.global_index = np.zeros(sh, dtype=int)
        self.requested_exposure_time = np.zeros(sh) * u.s
        self.run_mode = np.zeros(sh, dtype='S20')
        self.status = np.zeros(sh, dtype='S20')
        self.fpga_temp = np.zeros(sh) * u.adu
        self.fpga_vccint_voltage = np.zeros(sh) * u.adu
        self.fpga_vccaux_voltage = np.zeros(sh) * u.adu
        self.fpga_vccbram_voltage = np.zeros(sh) * u.adu
        self.adc_temp = np.zeros(tuple(sh) + (4, )) * u.adu
        self.optics = esis.optics.Optics()
        return self

    @property
    def slice_valid(self) -> slice:
        return slice(self.num_invalid_exposures, None)

    @property
    def intensity_valid(self) -> u.Quantity:
        return self.intensity[self.slice_valid]

    @property
    def time_valid(self) -> astropy.time.Time:
        return self.time[self.slice_valid]

    @property
    def time_index_valid(self):
        return self.time_index[self.slice_valid]

    @property
    def time_mission_start(self) -> astropy.time.Time:
        # return self.time_exp_start[0].min()
        return self.trajectory.time_start

    @property
    def time_shutter_open(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.shutter_door_open.time_mission

    @property
    def time_shutter_close(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.shutter_door_close.time_mission

    @property
    def time_rlg_enable(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.sparcs_rlg_enable.time_mission

    @property
    def time_rlg_disable(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.sparcs_rlg_disable.time_mission

    @property
    def time_parachute_deploy(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.parachute_deploy.time_mission

    def _calc_closest_index(
            self,
            t: astropy.time.Time,
            times: typ.Optional[astropy.time.Time] = None,
    ) -> int:
        if times is None:
            times = self.time_optimized
        dt = times - t
        return np.median(np.argmin(np.abs(dt.value), axis=0)).astype(int)

    @property
    def index_dark_up_first(self) -> int:
        return self.num_invalid_exposures

    @property
    def index_dark_down_last(self) -> int:
        return self.num_times - 1

    def _index_signal_first(self, time: astropy.time.Time) -> int:
        return self._calc_closest_index(t=self.time_rlg_enable, times=time) + 1

    def _index_signal_last(self, time: astropy.time.Time) -> int:
        return self._calc_closest_index(t=self.time_rlg_disable, times=time) - 1

    def _slice_signal(self, time: astropy.time.Time) -> slice:
        return slice(self._index_signal_first(time=time), self._index_signal_last(time=time) + 1)

    @property
    def bias(self) -> u.Quantity:
        s1, s2 = [slice(None)] * self.axis.ndim, [slice(None)] * self.axis.ndim
        s1[self.axis.x] = slice(self.num_ignored_bias_columns + 1, self.optics.detector.npix_blank)
        s2[self.axis.x] = slice(~(self.optics.detector.npix_blank - 1), ~(self.num_ignored_bias_columns - 1))
        blank_pix = 2 * [s1] + 2 * [s2]
        quadrants = self.optics.detector.quadrants
        bias_shape = (self.shape[self.axis.time], self.shape[self.axis.channel], len(quadrants))
        bias = np.empty(bias_shape) << self.intensity.unit
        for q in range(len(quadrants)):
            data_quadrant = self.intensity[(...,) + quadrants[q]]
            a = data_quadrant[tuple(blank_pix[q])]
            # bias[..., q] = np.median(a=a, axis=self.axis.xy)
            bias[..., q] = scipy.stats.trim_mean(
                a=a.reshape(a.shape[:~1] + (-1,)),
                proportiontocut=0.25,
                axis=~0
            ) << a.unit
        return bias

    @property
    def intensity_nobias(self) -> u.Quantity:
        """
        Returns
        -------
        Bias subtracted data
        """
        if self._intensity_nobias is None:
            intensity_nobias = self.intensity.copy()
            quadrants = self.optics.detector.quadrants
            bias = self.bias
            for q in range(len(quadrants)):
                intensity_nobias[(...,) + quadrants[q]] -= bias[..., q, np.newaxis, np.newaxis]
            self._intensity_nobias = intensity_nobias
        return self._intensity_nobias

    def _calc_dark(self, darks: u.Quantity) -> u.Quantity:
        return scipy.stats.trim_mean(a=darks, proportiontocut=0.25, axis=self.axis.time) << darks.unit
        # return np.median(darks, axis=self.axis.time)

    @classmethod
    def _remove_dark(cls, intensity: u.Quantity, master_dark: u.Quantity) -> u.Quantity:
        return intensity - master_dark

    @classmethod
    def _calc_stray_light_avg(cls, intensity: u.Quantity, num_pixels: int = 100):
        slice_lower = [slice(None)] * cls.axis.ndim
        slice_upper = [slice(None)] * cls.axis.ndim

        slice_lower[cls.axis.y] = slice(None, num_pixels)
        slice_upper[cls.axis.y] = slice(~(num_pixels - 1), None)

        strip_lower = intensity[tuple(slice_lower)]
        strip_upper = intensity[tuple(slice_upper)]

        proportion = 0.3

        strip_lower_avg = scipy.stats.trim_mean(
            a=strip_lower.reshape(strip_lower.shape[:~1] + (-1,)),
            proportiontocut=proportion,
            axis=~0,
        ) << strip_lower.unit
        strip_upper_avg = scipy.stats.trim_mean(
            a=strip_upper.reshape(strip_upper.shape[:~1] + (-1,)),
            proportiontocut=proportion,
            axis=~0,
        ) << strip_upper.unit

        return np.where(
            strip_lower_avg.sum(cls.axis.time) < strip_upper_avg.sum(cls.axis.time),
            strip_lower_avg,
            strip_upper_avg,
        )

    @property
    def intensity_electrons_nostray_prelim(self) -> u.Quantity:
        if self._intensity_electrons_prelim is None:
            index_dark_up_first = self.index_dark_up_first
            index_dark_up_last = index_dark_up_first + 5
            index_dark_down_last = self.index_dark_down_last
            index_dark_down_first = index_dark_down_last - 5
            darks_up = self.intensity_nobias[index_dark_up_first:index_dark_up_last + 1]
            darks_down = self.intensity_nobias[index_dark_down_first:index_dark_down_last + 1]
            darks = np.concatenate([darks_up, darks_down])
            dark = self._calc_dark(darks=darks)
            intensity_nobias_nodark = self._remove_dark(intensity=self.intensity_nobias, master_dark=dark)
            intensity_nobias_nodark_active = self.optics.detector.remove_inactive_pixels(intensity_nobias_nodark)
            intensity_electrons = self.optics.detector.convert_adu_to_electrons(intensity_nobias_nodark_active)
            stray_light = self._calc_stray_light_avg(intensity_electrons)
            intensity_electrons = intensity_electrons - stray_light[..., np.newaxis, np.newaxis]
            self._intensity_electrons_prelim = intensity_electrons
        return self._intensity_electrons_prelim

    @classmethod
    def _calc_intensity_avg(cls, intensity: u.Quantity) -> u.Quantity:
        return scipy.stats.trim_mean(
            intensity.reshape(intensity.shape[:~1] + (-1, )),
            proportiontocut=0.25,
            axis=~0,
        ) << intensity.unit

    @classmethod
    def _calc_atmosphere_transmission(
            cls,
            intensity_avg: u.Quantity,
            altitude: u.Quantity,
            sun_zenith_angle: u.Quantity,
    ) -> kgpy.atmosphere.TransmissionBates:
        return kgpy.atmosphere.TransmissionBates.from_data_fit(
            observer_height=altitude,
            zenith_angle=sun_zenith_angle,
            intensity_observed=intensity_avg,
            density_base=0.001 * u.g / u.cm**3,
            absorption_coefficient_bounds=[0.0001, 0.01] * u.cm**2 / u.g,
            particle_mass=16 * u.cds.mp,
            radius_base=120 * u.km,
            scale_height_bounds=[5, 200] * u.km,
            temperature_base=355 * u.K,
            temperature_infinity_bounds=[355, 2000] * u.K,
        )

    @property
    def transmission_atmosphere_model(self) -> kgpy.atmosphere.TransmissionBates:
        intensity_avg = self._calc_intensity_avg(self.intensity_electrons_nostray_prelim)
        slice_optimize = self._slice_optimize(intensity_avg)
        return self._calc_atmosphere_transmission(
                intensity_avg=intensity_avg[slice_optimize],
                altitude=self.altitude[slice_optimize],
                sun_zenith_angle=self.sun_zenith_angle[slice_optimize],
            )

    def _slice_optimize(self, intensity_avg: u.Quantity) -> np.array:
        intensity_avg_norm = intensity_avg / intensity_avg.max(self.axis.time)
        return intensity_avg_norm.mean(self.axis.channel) >= 0.25

    def _calc_offset_optimized(self) -> u.Quantity:

        intensity_avg = self._calc_intensity_avg(self.intensity_electrons_nostray_prelim)
        intensity_avg = intensity_avg / intensity_avg.max(self.axis.time)

        slice_optimize = self._slice_optimize(intensity_avg)

        def factory(
                params: np.ndarray,
        ) -> u.Quantity:
            return params[0] * u.s

        def objective(params: np.ndarray):
            time_offset = factory(params)
            time = self.time + time_offset
            altitude = self.trajectory.altitude_interp(t=time)
            sun_zenith_angle = self.trajectory.sun_zenith_angle_interp(t=time)
            # slice_optimize = self._slice_signal(time)
            transmission_model = self._calc_atmosphere_transmission(
                intensity_avg=intensity_avg[slice_optimize],
                altitude=altitude[slice_optimize],
                sun_zenith_angle=sun_zenith_angle[slice_optimize],
            )
            transmission = transmission_model(
                radius=altitude,
                zenith_angle=sun_zenith_angle,
            )
            intensity_corrected = intensity_avg / transmission

            # intensity_corrected = scipy.intensity_avg.detrend(
            #     data=intensity_corrected[slice_optimize],
            #     axis=0,
            # ) << intensity_corrected.unit
            intensity_corrected = intensity_corrected[slice_optimize]

            value = np.sqrt(np.mean(np.square(np.std(intensity_corrected, axis=0))))
            # intensity_normalized = intensity_avg / intensity_avg.max(self.axis.time)
            # intensity_normalized = intensity_normalized * transmission.max(self.axis.time)
            # value = np.sqrt(np.mean(np.square(intensity_normalized[slice_optimize] - transmission[slice_optimize])))
            # print(time_offset)
            # print(transmission_model)
            # print(value)
            # print()
            return value.value

        time_bound = self.exposure_length.mean() / 2
        params_optimized = scipy.optimize.brute(
            func=objective,
            Ns=21,
            ranges=[
                [-time_bound.value, time_bound.value],
            ],
            # finish=None,
        )

        # print(params_optimized)

        return factory(params_optimized[np.newaxis])

    @property
    def offset_optimized(self) -> u.Quantity:
        if self._offset_optimized is None:
            self._offset_optimized = self._calc_offset_optimized()
        return self._offset_optimized

    @property
    def time_optimized(self) -> astropy.time.Time:
        return self.time + self.offset_optimized

    @property
    def altitude(self) -> u.Quantity:
        return self.trajectory.altitude_interp(self.time_optimized)

    @property
    def sun_zenith_angle(self) -> u.Quantity:
        return self.trajectory.sun_zenith_angle_interp(self.time_optimized)

    @property
    def index_signal_first(self) -> int:
        return self._index_signal_first(time=self.time_optimized)

    @property
    def index_signal_last(self) -> int:
        return self._index_signal_last(time=self.time_optimized)

    @property
    def slice_signal(self) -> slice:
        return self._slice_signal(time=self.time_optimized)

    @property
    def index_apogee(self) -> int:
        return self._calc_closest_index(self.trajectory.time_apogee)

    @property
    def time_apogee(self) -> astropy.time.Time:
        return self.time_optimized[self.index_apogee]

    @property
    def slice_upleg(self) -> slice:
        return slice(None, self.index_apogee)

    @property
    def slice_downleg(self) -> slice:
        return slice(self.index_apogee, None)

    @property
    def index_dark_up_last(self) -> int:
        return self._calc_closest_index(self.time_shutter_open) - 1

    @property
    def index_dark_down_first(self) -> int:
        return self._calc_closest_index(self.time_parachute_deploy)

    @property
    def darks_up(self):
        return self.intensity_nobias[self.index_dark_up_first:self.index_dark_up_last + 1]

    @property
    def darks_down(self):
        return self.intensity_nobias[self.index_dark_down_first:self.index_dark_down_last + 1]

    @property
    def darks(self):
        return np.concatenate([self.darks_up, self.darks_down])

    @property
    def dark(self):
        return self._calc_dark(self.darks)

    @property
    def intensity_nobias_nodark(self) -> u.Quantity:
        return self._remove_dark(self.intensity_nobias, self.dark)

    @property
    def intensity_nobias_nodark_active(self):
        return self.optics.detector.remove_inactive_pixels(self.intensity_nobias_nodark)

    @property
    def intensity_electrons(self) -> u.Quantity:
        return self.optics.detector.convert_adu_to_electrons(self.intensity_nobias_nodark_active)

    @property
    def stray_light_avg(self) -> u.Quantity:
        if self._stray_light_avg is None:
            self._stray_light_avg = self._calc_stray_light_avg(self.intensity_electrons)
        return self._stray_light_avg

    @property
    def intensity_electrons_nostray(self) -> u.Quantity:
        return self.intensity_electrons - self.stray_light_avg[..., np.newaxis, np.newaxis]

    @property
    def intensity_electrons_nostray_avg(self) -> u.Quantity:
        return self._calc_intensity_avg(self.intensity_electrons_nostray)

    @property
    def transmission_atmosphere(self) -> u.Quantity:
        return self.transmission_atmosphere_model(
            radius=self.altitude,
            zenith_angle=self.sun_zenith_angle,
        )

    @property
    def intensity_electrons_nostray_noatm(self) -> u.Quantity:
        return self.intensity_electrons_nostray / self.transmission_atmosphere[..., np.newaxis, np.newaxis]

    @property
    def intensity_electrons_nostray_noatm_avg(self) -> u.Quantity:
        return self._calc_intensity_avg(self.intensity_electrons_nostray_noatm)

    @property
    def _time_plot_grid(self):
        return super()._time_plot_grid[1:]
    
    @property
    def _index_plot_grid(self):
        return super()._index_plot_grid[1:]

    def plot_intensity_nobias_mean(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        return self.plot_quantity_vs_index(
            ax=ax,
            a=self.intensity_nobias.mean(self.axis.xy),
            a_name='mean signal',
        )

    def plot_intensity_nobias_nodark_active_mean(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        return self.plot_quantity_vs_index(
            ax=ax,
            a=self.intensity_nobias_nodark_active.mean(self.axis.xy),
            a_name='mean signal',
        )

    def plot_intensity_electrons_mean(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        return self.plot_quantity_vs_index(
            ax=ax,
            a=self.intensity_electrons.mean(self.axis.xy),
            a_name='mean signal',
        )

    # def plot_intensity_derivative(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
    #     ax2, lines = self.plot_quantity_vs_index(
    #         ax=ax,
    #         a=self.intensity_derivative,
    #         a_name='signal derivative',
    #     )
    #     start_line = ax.axvline(
    #         x=self.time[self._index_deriv_max, 0].to_datetime(),
    #         color='black',
    #         label='max derivative',
    #         linestyle='--',
    #     )
    #     end_line = ax.axvline(
    #         x=self.time[self._index_deriv_min, 0].to_datetime(),
    #         color='black',
    #         label='min derivative',
    #         linestyle='--',
    #     )
    #     lines.append(start_line)
    #     lines.append(end_line)
    #     return ax2, lines

    def plot_fpga_temp(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        return self.plot_quantity_vs_index(a=self.fpga_temp, a_name='FPGA temp.', ax=ax, )

    def plot_fpga_vccint(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        return self.plot_quantity_vs_index(ax=ax, a=self.fpga_vccint_voltage, a_name='FPGA VCCint', )

    def plot_fpga_vccaux(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        return self.plot_quantity_vs_index(ax=ax, a=self.fpga_vccaux_voltage, a_name='FPGA VCCaux', )

    def plot_fpga_vccbram(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        return self.plot_quantity_vs_index(ax=ax, a=self.fpga_vccbram_voltage, a_name='FPGA BRAM', )

    def plot_adc_temperature(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        self.plot_quantity_vs_index(a=self.adc_temp[..., 0], a_name='ADC temp 1', ax=ax)
        self.plot_quantity_vs_index(a=self.adc_temp[..., 1], a_name='ADC temp 2', ax=ax)
        self.plot_quantity_vs_index(a=self.adc_temp[..., 2], a_name='ADC temp 3', ax=ax)
        self.plot_quantity_vs_index(a=self.adc_temp[..., 3], a_name='ADC temp 4', ax=ax)
        return None

    def plot_bias(self, ax: plt.Axes, ) -> typ.Tuple[plt.Axes, typ.List[plt.Line2D]]:
        bias = self.bias.mean(axis=~0)
        return self.plot_quantity_vs_index(ax=ax, a=bias, a_name='bias', )
        # num_quadrants = bias.shape[~0]
        # for q in range(num_quadrants):
        #     name = 'bias, q' + str(q)
        #     ax = self.plot_quantity_vs_index(a=bias[..., q], a_name=name, ax=ax, legend_ncol=num_quadrants // 2)
        # return ax

    def plot_dark_up_span(self, ax: plt.Axes) -> plt.Line2D:
        return ax.axvspan(
            xmin=self.time_exp_start[self.index_dark_up_first, 0].to_datetime(),
            xmax=self.time_exp_end[self.index_dark_up_last, 0].to_datetime(),
            alpha=0.3,
            color='gray',
            label='upleg dark frames',
        )

    def plot_dark_down_span(self, ax: plt.Axes) -> plt.Line2D:
        return ax.axvspan(
            xmin=self.time_exp_start[self.index_dark_down_first, 0].to_datetime(),
            xmax=self.time_exp_end[self.index_dark_down_last, 0].to_datetime(),
            alpha=0.4,
            color='gray',
            label='downleg dark frames',
        )

    def plot_signal_span(self, ax: plt.Axes) -> plt.Line2D:
        return ax.axvspan(
            xmin=self.time_exp_start[self.index_signal_first, 0].to_datetime(),
            xmax=self.time_exp_end[self.index_signal_last, 0].to_datetime(),
            alpha=0.3,
            color='green',
            label='light frames',
        )

    def plot_dark_spans(self, ax: plt.Axes):
        self.plot_dark_up_span(ax=ax)
        self.plot_dark_down_span(ax=ax)

    def plot_dark(self, axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None) -> typ.MutableSequence[plt.Axes]:
        axs[0].figure.suptitle('Median dark images')
        return self.plot_time(images=self.dark, image_names=self.channel_labels, axs=axs, )

    def plot_exposure_stats_vs_index(
            self, axs: typ.Sequence[plt.Axes],
    ) -> typ.Sequence[plt.Axes]:
        self.plot_intensity_electrons_mean(ax=axs[0])
        self.plot_exposure_length(ax=axs[1])
        self.plot_bias(ax=axs[2])

        for ax in axs[:~0]:
            ax.set_xlabel('')

        return axs

    def plot_altitude_and_signal_vs_time(
            self,
            ax: typ.Optional[plt.Axes],
            time: typ.Optional[astropy.time.Time] = None,
    ) -> plt.Axes:

        ax_twin = ax.twinx()
        ax_twin._get_lines.prop_cycler = ax._get_lines.prop_cycler

        if time is None:
            time = self.time_optimized

        signal = self.intensity_electrons_nostray_avg
        # self.plot_intensity_nobias_mean(ax=ax)
        # signal = np.median(self.intensity_nobias_nodark_active, axis=self.axis.xy)
        # signal = self.intensity_nobias_nodark_active.mean(self.axis.xy)
        signal = signal / signal.max(self.axis.time)

        self.plot_quantity_vs_index(
            a=signal,
            t=time,
            a_name='average intensity',
            ax=ax_twin,
        )
        self.trajectory.plot_altitude_vs_time(
            ax=ax,
            time_start=self.time_mission_start,
        )
        ax_twin.set_ylabel('normalized intensity')

        return ax_twin

    def plot_signal_vs_altitude(
            self,
            ax: plt.Axes,
            time: typ.Optional[astropy.time.Time] = None,
    ) -> plt.Axes:
        with astropy.visualization.quantity_support():
            if time is None:
                time = self.time_optimized

            altitude = self.trajectory.altitude_interp(time)
            # signal = self.intensity_electrons_avg
            signal = self._calc_intensity_avg(self.intensity_nobias)
            signal = signal / signal.max(0)
            apogee_index = self._calc_closest_index(self.trajectory.time_apogee, times=time)
            altitude_up, altitude_down = altitude[:apogee_index + 1], altitude[apogee_index:]
            signal_up, signal_down = signal[:apogee_index + 1], signal[apogee_index:]
            # signal = self.intensity_electrons.mean(self.axis.xy)
            # signal = np.median(self.intensity_electrons, axis=self.axis.xy)
            # signal = np.mean(self.intensity_electrons[..., 256:-256, 1024 + 256:-256], axis=self.axis.xy)
            # signal = np.percentile(self.intensity_electrons, 75, axis=self.axis.xy)
            for i in range(self.num_channels):
                lines_up, = ax.plot(
                    altitude_up[..., i],
                    signal_up[..., i],
                    label=self.channel_labels[i] + ' average signal, upleg',
                )
                lines_down, = ax.plot(
                    altitude_down[..., i],
                    signal_down[..., i],
                    label=self.channel_labels[i] + ' average signal, downleg',
                    color=lines_up.get_color(),
                    linestyle='--',
                )

        return ax

    def plot_bias_subtraction_vs_index(
            self, axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
    ) -> typ.MutableSequence[plt.Axes]:
        if axs is None:
            fig, axs = plt.subplots(nrows=3)
        self.plot_intensity_mean_vs_time(ax=axs[0])
        self.plot_bias(ax=axs[1])
        self.plot_intensity_nobias_mean(ax=axs[2])
        return axs

    def plot_fpga_stats_vs_index(
            self, axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
    ) -> typ.MutableSequence[plt.Axes]:
        if axs is None:
            fig, axs = plt.subplots(nrows=5)
        self.plot_fpga_temp(ax=axs[0])
        self.plot_adc_temperature(ax=axs[1])
        self.plot_fpga_vccint(ax=axs[2])
        self.plot_fpga_vccaux(ax=axs[3])
        self.plot_fpga_vccbram(ax=axs[4])
        return axs

    def blink_intensity_nobias_nodark(
            self,
            time_index: int = 0,
            channel_index: int = 0,
            ax: typ.Optional[plt.Axes] = None,
            thresh_min: u.Quantity = 0.01 * u.percent,
            thresh_max: u.Quantity = 99.9 * u.percent,
            norm_gamma: float = 1,
            frame_interval: u.Quantity = 1 * u.s,
    ):
        frames = u.Quantity([
            self.intensity[time_index, channel_index],
            self.intensity_nobias[time_index, channel_index],
            self.intensity_nobias_nodark[time_index, channel_index],
        ])

        time = self.time_optimized[time_index, channel_index]
        chan = self.channel[channel_index]
        seq_index = self.time_index[time_index]

        base_title = time.to_value('iso') + ', frame ' + str(int(seq_index)) + ', channel ' + str(int(chan.value))

        frame_names = [
            base_title + '\n raw',
            base_title + '\n bias subtracted',
            base_title + '\n bias and dark subtracted',
        ]

        return self.animate_channel(
            images=frames, image_names=frame_names,
            ax=ax, thresh_min=thresh_min, thresh_max=thresh_max, norm_gamma=norm_gamma, frame_interval=frame_interval,
        )
