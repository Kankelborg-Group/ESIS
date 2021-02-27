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
import kgpy.obs
import kgpy.nsroc
import kgpy.observatories
import esis

__all__ = ['Level_0']


@dataclasses.dataclass
class Level_0(
    # kgpy.observatories.Obs,
    kgpy.obs.Image,
):
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
    detector: typ.Optional[esis.optics.Detector] = None
    trajectory: typ.Optional[kgpy.nsroc.Trajectory] = None
    timeline: typ.Optional[kgpy.nsroc.Timeline] = None
    caching: bool = False
    num_dark_safety_frames: int = 1
    num_ignored_bias_columns: int = 20

    def __post_init__(self):
        super().__post_init__()
        self.update()

    def update(self) -> typ.NoReturn:
        self._intensity_derivative = None
        self._intensity_nobias = None
        self._intensity_nobias_nodark = None
        self._darks_nobias = None

    @classmethod
    def from_directory(
            cls,
            directory: pathlib.Path,
            detector: esis.optics.Detector,
            trajectory: typ.Optional[kgpy.nsroc.Trajectory] = None,
            timeline: typ.Optional[kgpy.nsroc.Timeline] = None,
            caching: bool = False,
            num_dark_safety_frames: int = 1,
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
        # bad_exposures = 0
        num_exposures = len(fits_list) - bad_exposures

        hdu = astropy.io.fits.open(fits_list[0, 0])[0]
        self = cls.zeros((num_exposures, num_channels) + hdu.data.shape)
        # self.exposure_length = self.exposure_length * u.adu / u.s
        self.detector = detector
        self.trajectory = trajectory
        self.timeline = timeline
        self.caching = caching
        self.num_dark_safety_frames = num_dark_safety_frames

        for i in range(num_exposures):
            self.intensity_uncertainty[i] = detector.readout_noise_image(num_channels)
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
        self.detector = esis.optics.Detector()
        return self

    @property
    def time_start(self) -> astropy.time.Time:
        # return self.time_exp_start[0].min()
        return self.trajectory.time_start

    @property
    def altitude(self) -> u.Quantity:
        interpolator = scipy.interpolate.interp1d(
            x=self.trajectory.time.to_value('mjd'),
            y=self.trajectory.altitude,
            kind='quadratic',
        )
        t = self.time.to_value('mjd')
        return interpolator(t) << self.trajectory.altitude.unit
        # return interpolator((t[1:] + t[:~0]) / 2) << self.trajectory.altitude.unit

    @property
    def intensity_derivative(self) -> u.Quantity:
        if self._intensity_derivative is None:
            m = np.percentile(self.intensity, 99, axis=self.axis.xy)
            self._intensity_derivative = np.gradient(m, axis=self.axis.time)
        return self._intensity_derivative

    @property
    def _index_deriv_max(self):
        indices = np.argmax(self.intensity_derivative, axis=self.axis.time)
        return scipy.stats.mode(indices)[0][0]

    @property
    def _index_deriv_min(self):
        indices = np.argmin(self.intensity_derivative, axis=self.axis.time)
        return scipy.stats.mode(indices)[0][0]

    @property
    def index_signal_first(self) -> int:
        return self._index_deriv_max - self.num_dark_safety_frames

    @property
    def index_signal_last(self) -> int:
        return self._index_deriv_min + self.num_dark_safety_frames

    @property
    def signal_slice(self) -> slice:
        return slice(self.index_signal_first, self.index_signal_last + 1)

    @property
    def bias(self) -> u.Quantity:
        s1, s2 = [slice(None)] * self.axis.ndim, [slice(None)] * self.axis.ndim
        s1[self.axis.x] = slice(self.num_ignored_bias_columns + 1, self.detector.npix_blank)
        s2[self.axis.x] = slice(~(self.detector.npix_blank - 1), ~(self.num_ignored_bias_columns - 1))
        blank_pix = 2 * [s1] + 2 * [s2]
        quadrants = self.detector.quadrants
        bias = np.empty(
            (self.shape[self.axis.time], self.shape[self.axis.channel], len(quadrants))) << self.intensity.unit
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
        intensity_nobias = self._intensity_nobias
        if intensity_nobias is None:
            intensity_nobias = self.intensity.copy()
            quadrants = self.detector.quadrants
            bias = self.bias
            for q in range(len(quadrants)):
                intensity_nobias[(...,) + quadrants[q]] -= bias[..., q, None, None]
            if self.caching:
                self._intensity_nobias = intensity_nobias
        return intensity_nobias

    @property
    def darks_nobias(self) -> u.Quantity:
        if self._darks_nobias is None:
            intensity = self.intensity_nobias
            first_ind, last_ind = self.index_signal_first, self.index_signal_last + 1
            self._darks_nobias = np.concatenate([intensity[:first_ind], intensity[last_ind:]])
        return self._darks_nobias

    @property
    def dark_nobias(self) -> u.Quantity:
        return np.median(self.darks_nobias, axis=self.axis.time)

    @property
    def intensity_nobias_nodark(self) -> u.Quantity:
        intensity_nobias_nodark = self._intensity_nobias_nodark
        if intensity_nobias_nodark is None:
            intensity_nobias_nodark = self.intensity_nobias - self.dark_nobias
            if self.caching:
                self._intensity_nobias_nodark = intensity_nobias_nodark
        return intensity_nobias_nodark

    @property
    def intensity_nobias_nodark_active(self):
        return self.detector.remove_inactive_pixels(self.intensity_nobias_nodark)

    @property
    def intensity_signal(self) -> u.Quantity:
        return self.intensity_nobias_nodark_active[self.signal_slice]

    @property
    def time_signal(self) -> astropy.time.Time:
        return self.time[self.signal_slice]

    @property
    def requested_exposure_time_signal(self) -> u.Quantity:
        return self.requested_exposure_time[self.signal_slice]

    @property
    def time_index_signal(self) -> u.Quantity:
        return self.time_index[self.signal_slice]

    def plot_quantity_vs_index(
            self,
            a: u.Quantity,
            a_name: str = '',
            ax: typ.Optional[plt.Axes] = None,
            legend_ncol: int = 1,
            drawstyle: str = 'steps-mid',
    ) -> plt.Axes:
        ax = super().plot_quantity_vs_index(a=a, a_name=a_name, ax=ax, legend_ncol=legend_ncol, drawstyle=drawstyle)
        # with astropy.visualization.time_support(format='isot'):
        start_line = ax.axvline(
            x=self.time_exp_start[self.index_signal_first, 0].to_datetime(),
            color='black',
            label='signal start',
        )
        end_line = ax.axvline(
            x=self.time_exp_end[self.index_signal_last, 0].to_datetime(),
            color='black',
            label='signal end'
        )
        # ax.legend()
        # ax.figure.legend(handles=[start_line, end_line])
        return ax

    def plot_intensity_nobias_mean(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_index(
            a=self.intensity_nobias.mean(self.axis.xy), a_name='mean signal', ax=ax)

    def plot_intensity_derivative(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        ax = self.plot_quantity_vs_index(a=self.intensity_derivative, a_name='signal derivative', ax=ax)
        start_line = ax.axvline(
            x=self.time[self._index_deriv_max, 0].to_value('mjd'),
            color='black',
            label='max derivative',
            linestyle='--',
        )
        end_line = ax.axvline(
            x=self.time[self._index_deriv_min, 0].to_value('mjd'),
            color='black',
            label='min derivative',
            linestyle='--',
        )

    def plot_fpga_temp(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_index(a=self.fpga_temp, a_name='FPGA temp.', ax=ax)

    def plot_fpga_vccint(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_index(a=self.fpga_vccint_voltage, a_name='FPGA VCCint', ax=ax)

    def plot_fpga_vccaux(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_index(a=self.fpga_vccaux_voltage, a_name='FPGA VCCaux', ax=ax)

    def plot_fpga_vccbram(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_index(a=self.fpga_vccbram_voltage, a_name='FPGA BRAM', ax=ax)

    def plot_adc_temperature(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        ax = self.plot_quantity_vs_index(a=self.adc_temp_1, a_name='ADC temp 1', ax=ax)
        ax = self.plot_quantity_vs_index(a=self.adc_temp_2, a_name='ADC temp 2', ax=ax)
        ax = self.plot_quantity_vs_index(a=self.adc_temp_3, a_name='ADC temp 3', ax=ax)
        ax = self.plot_quantity_vs_index(a=self.adc_temp_4, a_name='ADC temp 4', ax=ax, legend_ncol=2)
        return ax

    def plot_bias(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        bias = self.bias.mean(axis=~0)
        ax = self.plot_quantity_vs_index(a=bias, a_name='bias', ax=ax)
        # num_quadrants = bias.shape[~0]
        # for q in range(num_quadrants):
        #     name = 'bias, q' + str(q)
        #     ax = self.plot_quantity_vs_index(a=bias[..., q], a_name=name, ax=ax, legend_ncol=num_quadrants // 2)
        return ax

    def plot_dark(self, axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None) -> typ.MutableSequence[plt.Axes]:
        axs[0].figure.suptitle('Median dark images')
        return self.plot_time(images=self.dark_nobias, image_names=self.channel_labels, axs=axs, )

    def plot_exposure_stats_vs_index(
            self, axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
    ) -> typ.MutableSequence[plt.Axes]:
        if axs is None:
            fig, axs = plt.subplots(nrows=4)

        for ax in axs[:~0]:
            ax.set_xlabel('')

        axs_2 = [ax.secondary_xaxis(
            location='top',
            functions=(self._time_to_index, self._index_to_time),
        ) for ax in axs]
        axs_2[0].set_xlabel('exposure index')
        for ax in axs_2[1:]:
            ax.set_xticklabels([])

        # self.trajectory.plot_altitude_vs_time(axs[0])
        self.plot_intensity_nobias_mean(ax=axs[0])
        self.plot_intensity_derivative(ax=axs[1])
        self.plot_exposure_length(ax=axs[2])
        self.plot_bias(ax=axs[3])
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
            # ax=ax,
            ax=ax_twin,
            # drawstyle='default',
        )
        self.trajectory.plot_altitude_vs_time(
            ax=ax,
            time_start=self.time_start,
            # ax=ax_twin,
        )
        ax_twin.set_ylabel('normalized intensity')

        # ax.legend()
        # ax.legend(loc='center right', bbox_to_anchor=(-0.1, 0.5))
        # self.timeline.plot(ax=ax, time_start=self.trajectory.time_start)

        # lines, labels = ax.get_legend_handles_labels()
        # lines_twin, labels_twin = ax_twin.get_legend_handles_labels()
        #
        # ax.legend(lines + lines_twin, labels + labels_twin)
        # ax_twin.get_legend().remove()

        return ax_twin

    def plot_signal_vs_altitude(
            self,
            ax: typ.Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()

        with astropy.visualization.quantity_support():
            ax.plot(
                self.altitude,
                self.intensity_nobias_nodark_active.mean(self.axis.xy),
                # drawstyle='steps',
            )
            ax.legend(['mean signal, ch' + str(i + 1) for i in range(self.num_channels)])

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
