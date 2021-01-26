from esis.data import level_3
import pathlib

if __name__ == '__main__':
    lev3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    wcs = lev3.observation[0,0].wcs.dropaxis(-1).dropaxis(-1)
    aia = lev3.to_aia_object()
    path = pathlib.Path(__file__).parent / 'aia_304/'
    aia.to_fits(path, label = 'AIA_304_Lev3_')