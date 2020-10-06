from esis.data import Level_0, Level_1
import esis.flight.optics
import astropy.units as u
import matplotlib.pyplot as plt

def generate_level1():
    optics = esis.flight.optics.as_measured()
    detector = optics.detector
    lev0 = Level_0.from_directory(esis.flight.raw_img_dir,detector)


    lev1 = Level_1.from_level_0(lev0, despike = True)
    lev1.intensity = lev1.intensity_photons(630 * u.AA)
    return lev1

if __name__ == "__main__":
    lev1 = generate_level1()
    lev1.to_pickle()
