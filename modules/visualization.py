import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from modules.mask_processing import remove_small_components

def visualize_masks(image_file, remove_components=True, min_size=8):
    """
    Lê o TIFF (que já contém B2,B3,B4,B8,SCL,probability,cloud_mask,shadow_mask),
    gera RGB (B4,B3,B2) e aplica operação morfológica e/ou remoção de componentes na nuvem e sombra.
    """
    with rasterio.open(image_file) as src:
        b4 = src.read(3).astype(np.float32)
        b3 = src.read(2).astype(np.float32)
        b2 = src.read(1).astype(np.float32)

        cloud_mask = src.read(7)
        shadow_mask = src.read(8)

    rgb = np.stack([b4, b3, b2], axis=-1)
    max_val = np.max(rgb)
    if max_val > 0:
        rgb /= max_val

    cloud_mask = (cloud_mask > 0).astype(np.uint8)
    shadow_mask = (shadow_mask > 0).astype(np.uint8)

    if remove_components:
        cloud_mask = remove_small_components(cloud_mask, min_size=min_size)
        shadow_mask = remove_small_components(shadow_mask, min_size=min_size)

    kernel = np.ones((3,3), np.uint8)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB (B4,B3,B2)')
    axes[0].axis('off')

    axes[1].imshow(cloud_mask, cmap='Reds')
    axes[1].set_title('Máscara de Nuvem (final)')
    axes[1].axis('off')

    axes[2].imshow(shadow_mask, cmap='Blues')
    axes[2].set_title('Máscara de Sombra (final)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
