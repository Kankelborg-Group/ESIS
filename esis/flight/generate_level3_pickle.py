from esis.data import level_1,level_3
from esis.flight.generate_level1_pickle import generate_level1
import numpy as np



def generate_level3(line = 'ov',overwrite = False):
    if level_1.Level_1.default_pickle_path().exists() == False or overwrite == True:
        print('Generating Despiked Level1 Object')
        lev1 = generate_level1()
        lev1.to_pickle()


    if line == 'ov':
        print('Generating OV Level3 Object')
        if level_3.ov_Level3_initial.is_file() == False or overwrite == True:
            lev3 = level_3.Level3.from_aia_level1()
            lev3.to_pickle(level_3.ov_Level3_initial)
        else:
            lev3 = level_3.Level3.from_pickle(level_3.ov_Level3_initial)
        if level_3.ov_Level3_updated.is_file() == False or overwrite == True:
            lev3_updated = lev3.update_internal_alignment()
            lev3_updated.to_pickle(level_3.ov_Level3_updated)
        else:
            lev3_updated = level_3.Level3.from_pickle(level_3.ov_Level3_updated)

        lev3_masked = lev3_updated.add_mask(line = 'mgx')
        lev3_masked.to_pickle(level_3.ov_Level3_masked)

        scales = np.array([0.43610222, 0.33961842, 0.38185936, 0.49515337]) # hard coded for speed, can be found
        vignetting_correction = lev3_masked.correct_vignetting(scale_factor=scales)
        lev3_masked.observation.data[...] /= vignetting_correction

        #Equalize intensity to favorite channel
        means = lev3_masked.masked_mean_normalization()
        my_favorite_channel = 1
        for i in lev3_masked.lev1_cameras:
            lev3_masked.observation.data[:,i,:,:] *= means[:,my_favorite_channel,:,:]/means[:,i,:,:]


        lev3_masked.to_pickle(level_3.ov_final_path)

if __name__ == "__main__":
    lev3 = generate_level3()