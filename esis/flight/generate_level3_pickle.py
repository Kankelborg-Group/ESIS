import numpy as np
from esis.data import level_1, level_3
import esis.flight
import pathlib
from kgpy import img


def generate_level3(line='ov', despike=True):


    lev1 = esis.flight.level_1()

    if despike:
        intensity_unit = lev1.intensity.unit
        print('Despiking data, this will take a while ...')
        intensity, mask, stats = img.spikes.identify_and_fix(
            data=lev1.intensity.value,
            axis=(0, 2, 3),
            percentile_threshold=(0, 99.9),
            poly_deg=1,
        )
        lev1.intensity = intensity << intensity_unit

    if line == 'ov':
        print('Generating OV Level_3 Object')

        lev3 = level_3.Level_3.from_aia_level1(lev_1 = lev1)
        lev3_updated = lev3.update_internal_alignment(lev1 = lev1)
        lev3_masked = lev3_updated.add_mask(line='disperse')

        scales = np.array([0.43610222, 0.33961842, 0.38185936, 0.49515337])  # hard coded for speed, can be found
        vignetting_correction = lev3_masked.correct_vignetting(scale_factor=scales)
        lev3_masked.observation.data[...] /= vignetting_correction

        # Equalize intensity to favorite channel
        means = lev3_masked.masked_mean_normalization()
        brightest_channel = 0
        for i in lev3_masked.lev1_cameras:
            lev3_masked.observation.data[:, i, :, :] *= means[:, brightest_channel, :, :] / means[:, i, :, :]

        if despike:
            lev3_masked.to_pickle(level_3.ov_final_path)
        else:
            lev3_masked.to_pickle(level_3.ov_final_path_spikes)


        # lev3.to_pickle(level_3.ov_Level3_initial)
        # lev3_updated.to_pickle(level_3.ov_Level3_updated)
        # lev3_masked.to_pickle(level_3.ov_Level3_masked)


if __name__ == "__main__":
    generate_level3(despike=False)
