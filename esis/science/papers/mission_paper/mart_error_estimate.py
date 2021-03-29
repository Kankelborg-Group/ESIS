from esis.data import level_3, level_4
import numpy as np
import matplotlib.pyplot as plt
from esis.flight import l3_events
import skimage.transform
import astropy.units as u
import kgpy.img.coalignment.image_coalignment as img_align
from esis.flight.generate_level1_pickle import generate_level1
import esis.flight


if __name__ == '__main__':
    event = l3_events.perfectx
    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)
    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    l1 = esis.flight.level1()

    l4_int = l4.integrated_intensity
    brightest_pix = np.unravel_index(l4_int.argmax(), l4_int.shape)

    loc = np.array([(brightest_pix[1],), (brightest_pix[2],)]).T
    l4_world_coords = l4.wcs_list[brightest_pix[0]].dropaxis(0).all_pix2world(loc, 0)
    l3_pix = l3.observation.wcs.dropaxis(-1).dropaxis(-1).all_world2pix(l4_world_coords, 0)
    print(l3_pix)

    lev1_transforms = img_align.TransformCube.from_pickle(l3.transformation_objects).transform_cube
    l1_coords = lev1_transforms[brightest_pix[0]][1].transform_coordinates(l3_pix)
    print(l1_coords)

    # fig, ax = plt.subplots()
    # ax.imshow(l3.observation.data[brightest_pix[0],1,...], origin='lower')
    # plt.show()

    # region = l3.observation.data[:, :, event.location[0], event.location[1]]
    # channels = [0,1,2,3]
    # angles = (np.arange(4) * 45 - 22.5 + -45) * u.deg
    # seq = 15
    # projections = []
    # for chan in channels:
    #     # projection = scipy.ndimage.rotate(np.pad(region[seq,i]*window,((pad,pad),(pad,pad))),angle,**rotation_kwargs)
    #     projection = skimage.transform.rotate(region[seq, chan],
    #                                           angles[chan].value)
    #     projections.append(projection)
    #     fig, ax = plt.subplots()
    #     ax.imshow(projection, origin='lower')
    #     plt.show()
    #
    # projections = np.array(projections)
    # projections = projections[None, :, :, :, None]
    # projections[projections < 0] = 0
