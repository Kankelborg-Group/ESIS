import numpy as np
from kgpy.img.coalignment import image_coalignment as img_align
from esis.data import level_3
if __name__ == '__main__':
    l3_initial = level_3.Level_3.from_pickle(level_3.ov_Level3_initial)
    init_transforms = img_align.TransformCube.from_pickle(l3_initial.transformation_objects)
    l3_update = level_3.Level_3.from_pickle(level_3.ov_Level3_updated)
    update_transforms = img_align.TransformCube.from_pickle(l3_update.transformation_objects)

    l3_data = l3_initial.observation.data
    l3_img = l3_data[0, 0]
    mean_mag_difs = []
    for i in range(l3_data.shape[0]):
        for j in range(l3_data.shape[1]):

            init_transform_obj = init_transforms.transform_cube[i][j]
            init_transform_obj.transform = img_align.modified_affine_to_quadratic(init_transform_obj.transform, init_transform_obj.origin)
            init_transform_obj.transform_func = img_align.quadratic_transform
            init_coords_transformed = init_transform_obj.transform_func(l3_img, init_transform_obj.transform, init_transform_obj.origin)

            update_transform_obj = update_transforms.transform_cube[i][j]
            update_coords_transformed = update_transform_obj.transform_func(l3_img, update_transform_obj.transform, update_transform_obj.origin)
            coord_difs = init_coords_transformed - update_coords_transformed
            trim=300
            coord_difs = coord_difs[...,trim:-trim,trim:-trim]
            mag_dif = np.sqrt(coord_difs[0]**2+coord_difs[1]**2)
            if mag_dif.mean() > 1:
                print(i,j)
                print(mag_dif.mean())
                print(init_transform_obj.transform)
                print(init_transform_obj.post_transform_translation)
                print(update_transform_obj.transform)
                print(update_transform_obj.post_transform_translation)

            mean_mag_difs.append(np.mean(mag_dif))

    mean_mag_difs = np.array(mean_mag_difs)
    print(np.median(mean_mag_difs))