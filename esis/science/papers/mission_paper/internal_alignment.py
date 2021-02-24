from esis.data import level_3
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects

plt.rcParams.update({'font.size': 9})

if __name__ == '__main__':

    lev_3 = level_3.Level_3.from_pickle(level_3.ov_Level3_initial)
    camera_combos = [(0, 1), (1, 2), (1, 3), (0, 2), (0, 3), (2, 3)]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    marker = ['.', '.', '.', '*', '*', '*']

    normalized_imgs = lev_3.observation.data
    normalized_imgs = (normalized_imgs - normalized_imgs.mean(axis=(-2, -1), keepdims=True)) / normalized_imgs.std(
        axis=(-2, -1), keepdims=True)

    list_cc = []
    for lev3_seq, lev1_seq in enumerate(lev_3.lev1_sequences):
        for i, j in camera_combos:
            cc = np.sum(normalized_imgs[lev3_seq, i] * normalized_imgs[lev3_seq, j]) / normalized_imgs[lev3_seq, i].size
            list_cc.append(cc)

    list_cc = np.array(list_cc)
    print(list_cc.mean())

    updated_lev_3 = level_3.Level_3.from_pickle(level_3.ov_Level3_updated)
    normalized_imgs = updated_lev_3.observation.data
    normalized_imgs = (normalized_imgs - normalized_imgs.mean(axis=(-2, -1), keepdims=True)) / normalized_imgs.std(
        axis=(-2, -1), keepdims=True)

    updated_list_cc = []
    for lev3_seq, lev1_seq in enumerate(updated_lev_3.lev1_sequences):
        for i, j in camera_combos:
            cc = np.sum(normalized_imgs[lev3_seq, i] * normalized_imgs[lev3_seq, j]) / normalized_imgs[lev3_seq, i].size
            updated_list_cc.append(cc)

    updated_list_cc = np.array(updated_list_cc)
    print(updated_list_cc.mean())

    shp = normalized_imgs.shape
    cc_ratio = np.array(updated_list_cc / list_cc)
    cc_ratio = cc_ratio.reshape(6, shp[0])
    print(cc_ratio.shape)

    fig, ax = plt.subplots()
    ax.set_ylabel('Cross-Correlation Ratio')
    ax.set_xlabel('Level-3 Image Sequence')
    for lev3_seq, lev1_seq in enumerate(updated_lev_3.lev1_sequences):
        for i, combo in enumerate(camera_combos):
            ax.plot(lev3_seq, cc_ratio[i, lev3_seq], marker=marker[i], ls='', color=colors[i], label=camera_combos[i])
            if lev3_seq == 0:
                ax.legend()
    ax.axhline(y=1, color='r')

    # ax.legend()

    fig.savefig(fig_path / 'internal_align.pdf')
    plt.show()
