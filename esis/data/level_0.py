import typing as typ
import pathlib
import dataclasses
import numpy as np
import astropy.io.fits
from kgpy.io import fits

from esis.data import data


@dataclasses.dataclass
class Level_0:

    data: typ.Optional[np.ndarray] = None
    times: typ.Optional[np.ndarray] = None
    cam_id: typ.Optional[np.ndarray] = None
    cam_sn: typ.Optional[np.ndarray] = None
    global_index: typ.Optional[np.ndarray] = None
    sequence_index: typ.Optional[np.ndarray] = None
    requested_exposure_time: typ.Optional[np.ndarray] = None
    measured_exposure_time: typ.Optional[np.ndarray] = None
    run_mode: typ.Optional[np.ndarray] = None
    status: typ.Optional[np.ndarray] = None
    fpga_temp: typ.Optional[np.ndarray] = None
    fpga_vccint_voltage: typ.Optional[np.ndarray] = None
    fpga_vccaux_voltage: typ.Optional[np.ndarray] = None
    fpga_vccbram_voltage: typ.Optional[np.ndarray] = None
    adc_temp_1: typ.Optional[np.ndarray] = None
    adc_temp_2: typ.Optional[np.ndarray] = None
    adc_temp_3: typ.Optional[np.ndarray] = None
    adc_temp_4: typ.Optional[np.ndarray] = None

    @classmethod
    def from_directory(cls, directory: pathlib.Path):
        fits_list = np.array(list(directory.glob('*.fit*')))
        fits_list.sort()
        channels = np.array([int(f.name[4]) for f in fits_list])
        channels = np.unique(channels)
        fits_list = fits_list.reshape((len(channels), -1))
        fits_list = fits_list.transpose()

        self = cls()

        for fits_sublist in fits_list:
            for fits_file in fits_sublist:
                hdu = astropy.io.fits.open(fits_file)




    @classmethod
    def from_path(cls, path: pathlib.Path):

        frame_paths = data.find_frames(path, data.num_channels)

        hdu = fits.load_hdu(frame_paths)

        return cls(
            fits.extract_data(hdu),
            fits.extract_times(hdu, 'IMG_TS'),
            fits.extract_header_value(hdu, 'CAM_ID'),
            fits.extract_header_value(hdu, 'CAM_SN'),
            fits.extract_header_value(hdu, 'IMG_ISN'),
            fits.extract_header_value(hdu, 'IMG_CNT'),
            fits.extract_header_value(hdu, 'IMG_EXP'),
            fits.extract_header_value(hdu, 'MEAS_EXP'),
            fits.extract_header_value(hdu, 'RUN_MODE'),
            fits.extract_header_value(hdu, 'IMG_STAT'),
            fits.extract_header_value(hdu, 'FPGATEMP'),
            fits.extract_header_value(hdu, 'FPGAVINT'),
            fits.extract_header_value(hdu, 'FPGAVAUX'),
            fits.extract_header_value(hdu, 'FPGAVBRM'),
            fits.extract_header_value(hdu, 'ADCTEMP1'),
            fits.extract_header_value(hdu, 'ADCTEMP2'),
            fits.extract_header_value(hdu, 'ADCTEMP3'),
            fits.extract_header_value(hdu, 'ADCTEMP4'),
        )
