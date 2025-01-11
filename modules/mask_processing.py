import numpy as np
import cv2
from scipy.ndimage import label, generate_binary_structure

def generate_cloud_shadow_masks_sentinel2(image, cloud_prob_threshold=50):
    """
    Retorna duas bandas binárias (cloud_mask, shadow_mask) combinando:
      - Banda SCL (Scene Classification)
      - Banda 'probability' (0-100) da coleção COPERNICUS/S2_CLOUD_PROBABILITY
    """
    scl = image.select('SCL')
    cloud_prob = image.select('probability')

    cloud_mask_scl = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    cloud_mask_prob = cloud_prob.gt(cloud_prob_threshold)
    final_cloud_mask = cloud_mask_scl.Or(cloud_mask_prob).rename("cloud_mask")

    shadow_mask = scl.eq(3).rename("shadow_mask")

    return final_cloud_mask, shadow_mask

def remove_small_components(mask_array, min_size=8):
    """
    Remove componentes (blobs) menores que 'min_size' pixels usando scipy.ndimage.label.
    'mask_array' é um np.array binário (0 ou 1).
    Retorna um np.array binário (0 ou 1) sem os componentes pequenos.
    """
    structure = generate_binary_structure(2, 2)
    labeled, ncomponents = label(mask_array, structure=structure)

    output = np.copy(mask_array)
    for comp_id in range(1, ncomponents + 1):
        component_mask = (labeled == comp_id)
        if np.sum(component_mask) < min_size:
            output[component_mask] = 0

    return output
