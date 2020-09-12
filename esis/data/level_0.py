import typing as typ
import pathlib
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.colors
import matplotlib.animation
import astropy.units as u
import astropy.time
import astropy.io.fits
import astropy.visualization
import scipy.stats
import esis


@dataclasses.dataclass
class Level_0:
    data: np.ndarray
    time: astropy.time.Time
    cam_id: u.Quantity
    cam_sn: np.ndarray
    global_index: np.ndarray
    sequence_index: np.ndarray
    requested_exposure_time: u.Quantity
    measured_exposure_time: u.Quantity
    run_mode: np.ndarray
    status: np.ndarray
    fpga_temp: u.Quantity
    fpga_vccint_voltage: u.Quantity
    fpga_vccaux_voltage: u.Quantity
    fpga_vccbram_voltage: u.Quantity
    adc_temp_1: u.Quantity
    adc_temp_2: u.Quantity
    adc_temp_3: u.Quantity
    adc_temp_4: u.Quantity
    detector: esis.optics.components.Detector

    @classmethod
    def from_directory(cls, directory: pathlib.Path, detector: esis.optics.components.Detector):
        fits_list = np.array(list(directory.glob('*.fit*')))
        fits_list.sort()

        channels = np.array([int(f.name[4]) for f in fits_list])
        channels = np.unique(channels)
        num_channels = len(channels)

        fits_list = fits_list.reshape((num_channels, -1))
        fits_list = fits_list.transpose()
        bad_exposures = 2
        num_exposures = len(fits_list) - bad_exposures

        hdu = astropy.io.fits.open(fits_list[0, 0])[0]
        self = cls.zeros((num_exposures, num_channels) + hdu.data.shape)
        self.detector = detector

        for i in range(num_exposures):
            for c in range(num_channels):
                hdu = astropy.io.fits.open(fits_list[bad_exposures + i, c])[0]
                header = hdu.header
                self.data[i, c] = hdu.data
                self.time[i, c] = astropy.time.Time(header['IMG_TS'])
                self.cam_id[i, c] = int(header['CAM_ID'][~0]) * u.chan
                self.cam_sn[i, c] = int(header['CAM_SN'])
                self.global_index[i, c] = int(header['IMG_ISN'])
                self.sequence_index[i, c] = int(header['IMG_CNT'])
                self.requested_exposure_time[i, c] = float(header['IMG_EXP']) * u.ms
                self.measured_exposure_time[i, c] = float(header['MEAS_EXP']) * u.ct
                self.run_mode[i, c] = header['RUN_MODE']
                self.status[i, c] = header['IMG_STAT']
                self.fpga_temp[i, c] = float(header['FPGATEMP']) * u.ct
                self.fpga_vccint_voltage[i, c] = float(header['FPGAVINT']) * u.ct
                self.fpga_vccaux_voltage[i, c] = float(header['FPGAVAUX']) * u.ct
                self.fpga_vccbram_voltage[i, c] = float(header['FPGAVBRM']) * u.ct
                self.adc_temp_1[i, c] = float(header['ADCTEMP1']) * u.ct
                self.adc_temp_2[i, c] = float(header['ADCTEMP2']) * u.ct
                self.adc_temp_3[i, c] = float(header['ADCTEMP3']) * u.ct
                self.adc_temp_4[i, c] = float(header['ADCTEMP4']) * u.ct


        return self

    @classmethod
    def zeros(cls, shape: typ.Sequence[int]):
        sh = shape[:2]
        return cls(
            data=np.zeros(shape),
            time=astropy.time.Time(np.zeros(sh), format='unix'),
            cam_id=np.zeros(sh, dtype=np.int) * u.chan,
            cam_sn=np.zeros(sh, dtype=np.int),
            global_index=np.zeros(sh, dtype=np.int),
            sequence_index=np.zeros(sh),
            requested_exposure_time=np.zeros(sh) * u.s,
            measured_exposure_time=np.zeros(sh) * u.ct,
            run_mode=np.zeros(sh, dtype='S20'),
            status=np.zeros(sh, dtype='S20'),
            fpga_temp=np.zeros(sh) * u.ct,
            fpga_vccint_voltage=np.zeros(sh) * u.ct,
            fpga_vccaux_voltage=np.zeros(sh) * u.ct,
            fpga_vccbram_voltage=np.zeros(sh) * u.ct,
            adc_temp_1=np.zeros(sh) * u.ct,
            adc_temp_2=np.zeros(sh) * u.ct,
            adc_temp_3=np.zeros(sh) * u.ct,
            adc_temp_4=np.zeros(sh) * u.ct,
            detector=esis.optics.components.Detector()
        )

    @property
    def signal_indices(self) -> typ.Tuple[int, int]:

        threshold = 99
        num_border_frames = 4

        m = np.percentile(self.data, threshold, axis=(-2, -1))
        g = np.gradient(m, axis=0)
        start_ind = np.argmax(g, axis=0) - num_border_frames
        end_ind = np.argmin(g, axis=0) + num_border_frames
        start_ind = scipy.stats.mode(start_ind)[0][0]
        end_ind = scipy.stats.mode(end_ind)[0][0]

        return start_ind, end_ind

    @property
    def data_nobias_active(self) -> u.Quantity:
        """
        Returns
        -------
        Bias subtracted data with inactive pixels removed
        """
        return self.detector.remove_inactive_pixels(self.data_nobias)

    @property
    def data_nobias(self) -> u.Quantity:
        """
        Returns
        -------
        Bias subtracted data
        """
        frames = self.data.copy()

        s = [slice(None)] * frames.ndim
        s[-1] = slice(self.detector.npix_blank, ~(self.detector.npix_blank - 1))
        s = tuple(s)

        quadrants = self.detector.quadrants
        # find a bias for each quadrant
        b_1 = np.median(frames[(...,) + quadrants[0]][..., 0:s[-1].start], axis=(-2, -1), keepdims=True)
        b_2 = np.median(frames[(...,) + quadrants[1]][..., 0:s[-1].start], axis=(-2, -1), keepdims=True)
        b_3 = np.median(frames[(...,) + quadrants[2]][..., s[-1].stop:], axis=(-2, -1), keepdims=True)
        b_4 = np.median(frames[(...,) + quadrants[3]][..., s[-1].stop:], axis=(-2, -1), keepdims=True)

        # subtract bias from each quadrant
        frames[(...,) + quadrants[0]] -= b_1
        frames[(...,) + quadrants[1]] -= b_2
        frames[(...,) + quadrants[2]] -= b_3
        frames[(...,) + quadrants[3]] -= b_4

        return frames

    @property
    def data_light_dark(self) -> typ.Tuple[u.Quantity, u.Quantity]:

        start_ind, end_ind = self.signal_indices
        frames = self.data_nobias_active

        return Level_0.organize_array(frames, start_ind, end_ind)

    @staticmethod
    def organize_array(frames: np.ndarray, start_ind: int, end_ind: int) -> typ.Tuple[np.ndarray, np.ndarray]:
        dark1 = frames[:start_ind, ...]
        signal = frames[start_ind:end_ind, ...]
        dark2 = frames[end_ind:, ...]
        dark2 = dark2[:dark1.shape[0]]

        darks = np.concatenate([dark1, dark2], axis=0)

        return signal, darks

    @property
    def data_final(self) -> u.Quantity:
        data, darks = self.data_light_dark
        data = data - np.median(darks, axis=0, keepdims=True)
        return np.flip(data, axis=-2)



    @property
    def num_channels(self) -> int:
        return self.data.shape[1]

    @property
    def channel_labels(self) -> typ.List[str]:
        return ['Ch' + str(int(c.value)) for c in self.cam_id[0]]

    def plot_frame(
            self,
            ax: typ.Optional[plt.Axes] = None,
            exposure_index: int = 0,
            channel_index: int = 0,
            percentile_max: float = 99.9,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        data = self.data[exposure_index, channel_index]
        img = ax.imshow(
            X=data,
            vmax=np.percentile(data, percentile_max),
            origin='lower',
        )
        ax.set_xlabel('detector $x$ (pix)')
        ax.set_ylabel('detextor $y$ (pix)')
        fig.colorbar(img, ax=ax, label='DN')
        return ax

    def animate_channel(
            self,
            ax: typ.Optional[plt.Axes],
            exposure_slice: typ.Optional[slice] = None,
            channel_index: int = 0,
            percentile_max: u.Quantity = 99.9 * u.percent,
            norm_gamma: float = 1,
            frame_interval: u.Quantity = 100 * u.ms,
    ) -> matplotlib.animation.FuncAnimation:

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if exposure_slice is None:
            exposure_slice = slice(None)

        data = self.data[exposure_slice, channel_index]
        time = self.time[exposure_slice, channel_index]
        ind = self.sequence_index[exposure_slice, channel_index]
        chan = self.cam_id[exposure_slice, channel_index]

        vmax = np.percentile(data, percentile_max.value)
        img = ax.imshow(
            X=data[0],
            norm=matplotlib.colors.PowerNorm(gamma=norm_gamma),
            vmax=vmax,
            origin='lower',
        )
        def title_text(i: int) -> str:
            return time[i].to_value('iso') + ', channel ' + str(int(chan[i].value)) + ', frame ' + str(int(ind[i]))

        title = ax.set_title(title_text(0))
        ax.set_xlabel('detector $x$ (pix)')
        ax.set_ylabel('detextor $y$ (pix)')
        fig.colorbar(img, ax=ax, label='DN')

        def func(i: int):
            img.set_data(data[i])
            title.set_text(title_text(i))

        return matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=data.shape[0],
            interval=frame_interval.to(u.ms).value,
        )

    def plot_quantity_vs_time(
            self,
            a: u.Quantity,
            a_name: str = '',
            ax: typ.Optional[plt.Axes] = None,
            legend_ncol: int = 1
    ):
        if ax is None:
            fig, ax = plt.subplots()
        with astropy.visualization.quantity_support():
            with astropy.visualization.time_support(format='iso'):
                for c in range(self.num_channels):
                    if c == 0:
                        color = None
                    else:
                        color = line[0].get_color()
                    line = ax.plot(
                        self.time[:, c],
                        a[:, c],
                        color=color,
                        linestyle=list(matplotlib.lines.lineStyles.keys())[c],
                        label=a_name + ', ' + self.channel_labels[c],
                    )
        ax.legend(fontsize='xx-small', ncol=legend_ncol)
        return ax

    def plot_total_intensity(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_time(a=self.data.sum((~1, ~0)), a_name='Total DN', ax=ax)

    def plot_exposure_time(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_time(a=self.requested_exposure_time, a_name='Requested exposure time', ax=ax)

    def plot_fpga_temp(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_time(a=self.fpga_temp, a_name='FPGA temp.', ax=ax)

    def plot_fpga_vccint(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_time(a=self.fpga_vccint_voltage, a_name='FPGA VCCint', ax=ax)

    def plot_fpga_vccaux(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_time(a=self.fpga_vccaux_voltage, a_name='FPGA VCCaux', ax=ax)

    def plot_fpga_vccbram(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        return self.plot_quantity_vs_time(a=self.fpga_vccbram_voltage, a_name='FPGA BRAM', ax=ax)

    def plot_adc_temperature(self, ax: typ.Optional[plt.Axes] = None, ) -> plt.Axes:
        ax = self.plot_quantity_vs_time(a=self.adc_temp_1, a_name='ADC temp 1', ax=ax)
        ax = self.plot_quantity_vs_time(a=self.adc_temp_2, a_name='ADC temp 2', ax=ax)
        ax = self.plot_quantity_vs_time(a=self.adc_temp_3, a_name='ADC temp 3', ax=ax)
        ax = self.plot_quantity_vs_time(a=self.adc_temp_4, a_name='ADC temp 4', ax=ax, legend_ncol=4)
        return ax

    def plot_all_vs_time(
            self, axs: typ.Optional[typ.MutableSequence[plt.Axes]] = None,
    ) -> typ.MutableSequence[plt.Axes]:
        if axs is None:
            fig, axs = plt.subplots(nrows=7, sharex=True)
        self.plot_total_intensity(ax=axs[0])
        self.plot_exposure_time(ax=axs[1])
        self.plot_fpga_temp(ax=axs[2])
        self.plot_adc_temperature(ax=axs[3])
        self.plot_fpga_vccint(ax=axs[4])
        self.plot_fpga_vccaux(ax=axs[5])
        self.plot_fpga_vccbram(ax=axs[6])
        return axs
