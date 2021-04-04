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
import scipy.stats
import scipy.interpolate
import scipy.optimize
import kgpy.vector
import kgpy.model
import kgpy.obs
import kgpy.nsroc
import kgpy.observatories
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
    adc_temp_1: typ.Optional[u.Quantity] = None
    adc_temp_2: typ.Optional[u.Quantity] = None
    adc_temp_3: typ.Optional[u.Quantity] = None
    adc_temp_4: typ.Optional[u.Quantity] = None
    # detector: typ.Optional[esis.optics.Detector] = None
    optics: typ.Optional[esis.optics.Optics] = None
    trajectory_raw: typ.Optional[kgpy.nsroc.Trajectory] = None
    timeline: typ.Optional[kgpy.nsroc.Timeline] = None
    # num_dark_safety_frames: int = 1
    num_ignored_bias_columns: int = 20

    def __post_init__(self):
        super().__post_init__()
        self.update()

    def update(self) -> typ.NoReturn:
        self._intensity_derivative = None
        self._intensity_nobias = None
        self._darks_nobias = None
        self._trajectory = None

    @classmethod
    def from_directory(
            cls,
            directory: pathlib.Path,
            # detector: esis.optics.Detector,
            optics: typ.Optional[esis.optics.Optics] = None,
            trajectory_raw: typ.Optional[kgpy.nsroc.Trajectory] = None,
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
        bad_exposures = 2
        num_exposures = len(fits_list) - bad_exposures

        hdu = astropy.io.fits.open(fits_list[0, 0])[0]
        self = cls.zeros((num_exposures, num_channels) + hdu.data.shape)
        # self.detector = detector
        self.optics = optics
        self.trajectory_raw = trajectory_raw
        self.timeline = timeline
        # self.num_dark_safety_frames = num_dark_safety_frames

        for i in range(num_exposures):
            self.intensity_uncertainty[i] = optics.detector.readout_noise_image(num_channels)
            for c in range(num_channels):
                hdu = astropy.io.fits.open(fits_list[bad_exposures + i, c])[0]
                self.intensity[i, c] = hdu.data * u.adu
                self.time[i, c] = astropy.time.Time(hdu.header['IMG_TS'])
                self.exposure_length[i, c] = float(hdu.header['MEAS_EXP']) * 25 * u.ns
                if i == 0:
                    self.channel[c] = int(hdu.header['CAM_ID'][~0]) * u.chan
                if c == 0:
                    self.time_index[i] = int(hdu.header['IMG_CNT']) - bad_exposures
                self.cam_sn[i, c] = int(hdu.header['CAM_SN'])
                self.global_index[i, c] = int(hdu.header['IMG_ISN'])
                self.requested_exposure_time[i, c] = float(hdu.header['IMG_EXP']) * u.ms
                self.run_mode[i, c] = hdu.header['RUN_MODE']
                self.status[i, c] = hdu.header['IMG_STAT']
                self.fpga_temp[i, c] = float(hdu.header['FPGATEMP']) * u.adu
                self.fpga_vccint_voltage[i, c] = float(hdu.header['FPGAVINT']) * u.adu
                self.fpga_vccaux_voltage[i, c] = float(hdu.header['FPGAVAUX']) * u.adu
                self.fpga_vccbram_voltage[i, c] = float(hdu.header['FPGAVBRM']) * u.adu
                self.adc_temp_1[i, c] = float(hdu.header['ADCTEMP1']) * u.adu
                self.adc_temp_2[i, c] = float(hdu.header['ADCTEMP2']) * u.adu
                self.adc_temp_3[i, c] = float(hdu.header['ADCTEMP3']) * u.adu
                self.adc_temp_4[i, c] = float(hdu.header['ADCTEMP4']) * u.adu

        self.time = self.time - self.exposure_length / 2

        return self

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]) -> 'Level_0':
        sh = shape[:2]
        self = super().zeros(shape)  # type: Level_0
        self.cam_sn = np.zeros(sh, dtype=np.int)
        self.global_index = np.zeros(sh, dtype=np.int)
        self.requested_exposure_time = np.zeros(sh) * u.s
        self.run_mode = np.zeros(sh, dtype='S20')
        self.status = np.zeros(sh, dtype='S20')
        self.fpga_temp = np.zeros(sh) * u.adu
        self.fpga_vccint_voltage = np.zeros(sh) * u.adu
        self.fpga_vccaux_voltage = np.zeros(sh) * u.adu
        self.fpga_vccbram_voltage = np.zeros(sh) * u.adu
        self.adc_temp_1 = np.zeros(sh) * u.adu
        self.adc_temp_2 = np.zeros(sh) * u.adu
        self.adc_temp_3 = np.zeros(sh) * u.adu
        self.adc_temp_4 = np.zeros(sh) * u.adu
        self.optics = esis.optics.Optics()
        return self

    @property
    def time_mission_start(self) -> astropy.time.Time:
        # return self.time_exp_start[0].min()
        return self.trajectory.time_start

    @property
    def time_shutter_open(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.shutter_door_open.time_mission

    @property
    def time_rlg_enable(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.sparcs_rlg_enable.time_mission

    @property
    def time_rlg_disable(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.sparcs_rlg_disable.time_mission

    @property
    def time_shutter_close(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.shutter_door_close.time_mission

    @property
    def time_parachute_deploy(self) -> astropy.time.Time:
        return self.time_mission_start + self.timeline.parachute_deploy.time_mission

    def _calc_closest_index(self, t: astropy.time.Time) -> int:
        dt = self.time - t
        return np.median(np.argmin(np.abs(dt.value), axis=0)).astype(int)

    @property
    def trajectory(self) -> kgpy.nsroc.Trajectory:

        if self._trajectory is None:

            signal = np.mean(self.intensity_nobias, axis=self.axis.xy)
            # signal = np.mean(self.intensity_nobias[..., 256:-256, 1024 + 256:-256], axis=self.axis.xy)
            # signal = np.median(self.intensity_nobias[..., 1024:], axis=self.axis.xy)
            # signal = np.percentile(self.intensity_nobias, 75, axis=self.axis.xy)

            def objective(t: float):
                trajectory = self.trajectory_raw.copy()
                trajectory.time_start = trajectory.time_start + t * u.s
                altitude = trajectory.altitude_interp(self.time)
                apogee_index = self._calc_closest_index(trajectory.time_apogee)
                altitude_up, altitude_down = altitude[:apogee_index], altitude[apogee_index:]
                signal_up, signal_down = signal[:apogee_index], signal[apogee_index:]
                mask = altitude_up > 200 * u.km
                result = 0
                for i in range(self.num_channels):
                    signal_down_interp = scipy.interpolate.interp1d(altitude_down[..., i], signal_down[..., i])
                    mask_i = mask[..., i]
                    diff = signal_up[mask_i].value - signal_down_interp(altitude_up[mask_i])
                    result = result + np.mean(np.square(diff))
                result = np.sqrt(result / self.num_channels)
                return result

            bound = 4 * self.exposure_length.mean()

            time_offset, *_ = scipy.optimize.brute(
                func=objective,
                Ns=100,
                ranges=[[-bound.value, bound.value]],
            )

            self._trajectory = self.trajectory_raw.copy()
            self._trajectory.time_start = self._trajectory.time_start + time_offset * u.s
        return self._trajectory

    @property
    def altitude(self) -> u.Quantity:
        return self.trajectory.altitude_interp(self.time)

    # @property
    # def intensity_derivative(self) -> u.Quantity:
    #     if self._intensity_derivative is None:
    #         m = np.percentile(self.intensity, 99, axis=self.axis.xy)
    #         self._intensity_derivative = np.gradient(m, axis=self.axis.time)
    #     return self._intensity_derivative
    #
    # @property
    # def _index_deriv_max(self):
    #     indices = np.argmax(self.intensity_derivative, axis=self.axis.time)
    #     return scipy.stats.mode(indices)[0][0]
    #
    # @property
    # def _index_deriv_min(self):
    #     indices = np.argmin(self.intensity_derivative, axis=self.axis.time)
    #     return scipy.stats.mode(indices)[0][0]

    @property
    def index_dark_up_first(self) -> int:
        return 0

    @property
    def index_dark_up_last(self) -> int:
        return self._calc_closest_index(self.time_shutter_open) - 1

    @property
    def index_dark_down_first(self) -> int:
        return self._calc_closest_index(self.time_parachute_deploy)

    @property
    def index_dark_down_last(self) -> int:
        return self.num_times - 1

    @property
    def index_signal_first(self) -> int:
        return self._calc_closest_index(self.time_rlg_enable) + 1
        # return self._index_deriv_max - self.num_dark_safety_frames

    @property
    def index_signal_last(self) -> int:
        return self._calc_closest_index(self.time_rlg_disable) - 1
        # return self._index_deriv_min + self.num_dark_safety_frames

    @property
    def slice_signal(self) -> slice:
        return slice(self.index_signal_first, self.index_signal_last + 1)

    @property
    def index_apogee(self) -> int:
        return self._calc_closest_index(self.trajectory.time_apogee)

    @property
    def time_apogee(self) -> astropy.time.Time:
        return self.time[self.index_apogee]

    @property
    def slice_upleg(self) -> slice:
        return slice(None, self.index_apogee)

    @property
    def slice_downleg(self) -> slice:
        return slice(self.index_apogee, None)

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
            a = data_quadrant[blank_pix[q]]
            bias[..., q] = np.median(a=a, axis=self.axis.xy)
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
        return np.median(self.darks, axis=self.axis.time)

    @property
    def intensity_nobias_nodark(self) -> u.Quantity:
        return self.intensity_nobias - self.dark

    @property
    def intensity_nobias_nodark_active(self):
        return self.optics.detector.remove_inactive_pixels(self.intensity_nobias_nodark)

    @property
    def intensity_electrons(self) -> u.Quantity:
        return self.optics.detector.convert_adu_to_electrons(self.intensity_nobias_nodark_active)

    @property
    def intensity_signal(self) -> u.Quantity:
        return self.intensity_electrons[self.slice_signal]

    @property
    def time_signal(self) -> astropy.time.Time:
        return self.time[self.slice_signal]

    @property
    def exposure_length_signal(self) -> u.Quantity:
        return self.exposure_length[self.slice_signal]

    @property
    def time_index_signal(self) -> u.Quantity:
        return self.time_index[self.slice_signal]

    @property
    def absorption_atmosphere(self) -> kgpy.model.Logistic:
        altitude_max = self.altitude.max()
        signal = self.intensity_signal.mean(self.axis.xy)
        return kgpy.model.Logistic.from_data_fit(
            x=self.altitude[self.slice_signal],
            y=signal,
            amplitude_guess=signal.max(),
            offset_x_guess=altitude_max / 2,
            slope_guess=1 / altitude_max,
        )

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
        ax = self.plot_quantity_vs_index(a=self.adc_temp_1, a_name='ADC temp 1', ax=ax)
        ax = self.plot_quantity_vs_index(a=self.adc_temp_2, a_name='ADC temp 2', ax=ax)
        ax = self.plot_quantity_vs_index(a=self.adc_temp_3, a_name='ADC temp 3', ax=ax)
        ax = self.plot_quantity_vs_index(a=self.adc_temp_4, a_name='ADC temp 4', ax=ax, legend_ncol=2)
        return ax

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
            label='upleg darks',
        )

    def plot_dark_down_span(self, ax: plt.Axes) -> plt.Line2D:
        return ax.axvspan(
            xmin=self.time_exp_start[self.index_dark_down_first, 0].to_datetime(),
            xmax=self.time_exp_end[self.index_dark_down_last, 0].to_datetime(),
            alpha=0.3,
            color='gray',
            label='downleg darks',
        )

    def plot_signal_span(self, ax: plt.Axes) -> plt.Line2D:
        return ax.axvspan(
            xmin=self.time_exp_start[self.index_signal_first, 0].to_datetime(),
            xmax=self.time_exp_end[self.index_signal_last, 0].to_datetime(),
            alpha=0.3,
            color='green',
            label='signal',
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
    ) -> plt.Axes:

        ax_twin = ax.twinx()
        ax_twin._get_lines.prop_cycler = ax._get_lines.prop_cycler

        # self.plot_intensity_nobias_mean(ax=ax)
        signal = self.intensity_nobias_nodark_active.mean(self.axis.xy)
        signal = signal / signal.max(self.axis.time)

        self.plot_quantity_vs_index(
            a=signal,
            a_name='mean intensity',
            ax=ax_twin,
        )
        self.trajectory.plot_altitude_vs_time(
            ax=ax,
            time_start=self.time_mission_start,
        )
        ax_twin.set_ylabel('normalized intensity')

        return ax_twin

    def plot_signal_vs_altitude(self, ax: plt.Axes, plot_model: bool = True) -> plt.Axes:
        with astropy.visualization.quantity_support():
            altitude = self.altitude
            signal = self.intensity_electrons.mean(self.axis.xy)
            # signal = np.median(self.intensity_electrons[..., 1024:], axis=self.axis.xy)
            # signal = np.mean(self.intensity_electrons[..., 256:-256, 1024 + 256:-256], axis=self.axis.xy)
            # signal = np.percentile(self.intensity_electrons, 75, axis=self.axis.xy)
            absorption_model = self.absorption_atmosphere(altitude)
            if plot_model:
                ls = '-.'
            else:
                ls = 'default'
            for i in range(self.num_channels):
                lines_up, = ax.plot(
                    altitude[self.slice_upleg, i],
                    signal[self.slice_upleg, i],
                    label=self.channel_labels[i] + ' mean signal, upleg',
                    linestyle=ls,
                )
                lines_down, = ax.plot(
                    altitude[self.slice_downleg, i],
                    signal[self.slice_downleg, i],
                    label=self.channel_labels[i] + ' mean signal, downleg',
                    color=lines_up.get_color(),
                    linestyle='--',
                )
                if plot_model:
                    lines_model = ax.plot(
                        altitude[..., i],
                        absorption_model[..., i],
                        label=self.channel_labels[i] + ' modeled response',
                        color=lines_up.get_color(),
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

        time = self.time[time_index, channel_index]
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
