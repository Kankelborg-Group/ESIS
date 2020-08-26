from esis.data import level_1,level_3
from esis.flight.generate_level1_pickle import generate_level1


def generate_level3_pickle(line = 'ov'):
    if level_1.Level1.default_pickle_path().exists() is False:
        print('Generating Despiked Level1 Object')
        lev1 = generate_level1()
        lev1.to_pickle()
    else:

        if line is 'ov':
            lev3 = level_3.Level3.from_aia_level1()
            lev3.to_pickle(level_3.ov_Level3_initial)
            lev3_updated = lev3.update_internal_alignment()
            lev3_updated.to_pickle(level_3.ov_Level3_updated)
            lev3_updated.add_mask(line = 'mgx')
            lev3_updated.to_pickle(level_3.ov_Level3_masked)

