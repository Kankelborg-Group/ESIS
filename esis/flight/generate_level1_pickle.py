from esis.data import data, level_0, level_1

def generate_level1():
    lev0 = level_0.Level0.from_path(data.raw_path)
    lev1 = level_1.Level1.from_level_0(lev0,despike = True)

    return lev1

if __name__ == "__main__":
    lev1 = generate_level1()
    lev1.to_pickle()
