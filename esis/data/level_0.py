import typing
import pathlib
import dataclasses
import numpy as np
from kgpy.io import fits

from esis.data import data


@dataclasses.dataclass
class Level0:

    data: typing.Optional[np.ndarray] = None
    times: typing.Optional[np.ndarray] = None
    cam_id: typing.Optional[np.ndarray] = None
    cam_sn: typing.Optional[np.ndarray] = None
    global_index: typing.Optional[np.ndarray] = None
    sequence_index: typing.Optional[np.ndarray] = None
    requested_exposure_time: typing.Optional[np.ndarray] = None
    measured_exposure_time: typing.Optional[np.ndarray] = None
    run_mode: typing.Optional[np.ndarray] = None
    status: typing.Optional[np.ndarray] = None
    fpga_temp: typing.Optional[np.ndarray] = None
    fpga_vccint_voltage: typing.Optional[np.ndarray] = None
    fpga_vccaux_voltage: typing.Optional[np.ndarray] = None
    fpga_vccbram_voltage: typing.Optional[np.ndarray] = None
    adc_temp_1: typing.Optional[np.ndarray] = None
    adc_temp_2: typing.Optional[np.ndarray] = None
    adc_temp_3: typing.Optional[np.ndarray] = None
    adc_temp_4: typing.Optional[np.ndarray] = None

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
