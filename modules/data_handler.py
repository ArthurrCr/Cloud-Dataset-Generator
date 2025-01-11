import os
import geemap

def create_directories(base_dir='../data/dataset_sentinel/'):
    """
    Cria os diretórios necessários para armazenar os dados e máscaras.
    """
    os.makedirs(base_dir, exist_ok=True)
    mask_dir = os.path.join(base_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    return base_dir, mask_dir

def download_sentinel_image_with_masks(image, aoi, output_dir, image_id, cloud_prob_threshold=50):
    """
    - Seleciona B2,B3,B4,B8,SCL, + probability (já deve estar presente no 'image' após join).
    - Cria as bandas de máscara (cloud_mask, shadow_mask).
    - Exporta tudo para um único TIFF.
    - Retorna o caminho do arquivo salvo.
    """
    base_bands = ['B2','B3','B4','B8','SCL','probability']
    image = image.select(base_bands)

    from modules.mask_processing import generate_cloud_shadow_masks_sentinel2

    cloud_mask, shadow_mask = generate_cloud_shadow_masks_sentinel2(image, cloud_prob_threshold)
    image_with_masks = image.addBands([cloud_mask, shadow_mask])

    filename = os.path.join(output_dir, f'sentinel_{image_id}.tif')
    filename = os.path.normpath(filename)

    geemap.ee_export_image(
        image_with_masks,
        filename=filename,
        scale=10,
        region=aoi.getInfo()['coordinates'],
        file_per_band=False
    )

    return filename
