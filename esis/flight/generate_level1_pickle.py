from esis.data import data, level_0, level_1
import esis.flight.optics
def generate_level1():
    lev0 = level_0.Level0.from_path(esis.flight.raw_img_dir)
    optics = esis.flight.optics.as_measured()
    detector = optics.components.detector

    lev1 = level_1.Level_1.from_level_0(lev0, detector, despike = True)

    return lev1

if __name__ == "__main__":
    lev1 = generate_level1()
    lev1.to_pickle()
