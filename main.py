import ee
from modules.ee_utils import authenticate_ee, merge_s2_and_cloud_prob
from modules.data_handler import create_directories, download_sentinel_image_with_masks
from modules.visualization import visualize_masks
import os

def main():
    # Autentica e inicializa o Earth Engine
    authenticate_ee()

    # Cria os diretórios necessários
    dataset_dir, mask_dir = create_directories()

    # Define as AOIs
    aois = [
        ee.Geometry.Rectangle([-47.95, -15.83, -47.91, -15.79]),  # Brasília (~4km x ~4km)
        ee.Geometry.Rectangle([-43.22, -22.95, -43.18, -22.91]),  # Rio de Janeiro
        ee.Geometry.Rectangle([-46.75, -23.60, -46.71, -23.56]),  # São Paulo
    ]

    for idx, aoi in enumerate(aois):
        print(f"\nProcessando AOI {idx+1}/{len(aois)}: {aoi.getInfo()}")
        
        # 1) Coleção Sentinel-2 SR
        s2_sr_col = (
            ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(aoi)
            .filterDate('2021-01-01', '2021-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))  # Filtro inicial
        )

        # 2) Coleção Probability
        s2_cp_col = (
            ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(aoi)
            .filterDate('2021-01-01', '2021-12-31')
        )

        # 3) Faz o join das coleções
        merged_collection = merge_s2_and_cloud_prob(s2_sr_col, s2_cp_col)

        # Lista de imagens
        merged_list = merged_collection.toList(merged_collection.size())
        n_images = merged_list.size().getInfo()
        print(f"  -> Total de imagens pareadas = {n_images}")

        for img_idx in range(n_images):
            combined_image = ee.Image(merged_list.get(img_idx))
            # Tenta obter ID
            image_id = combined_image.get('PRODUCT_ID').getInfo()
            if not image_id:
                image_id = combined_image.get('system:index').getInfo()

            print(f"     -> Processando imagem {img_idx+1}/{n_images}, ID: {image_id}")

            try:
                # Exporta a imagem + máscaras em uma só vez
                image_file = download_sentinel_image_with_masks(
                    image=combined_image,
                    aoi=aoi,
                    output_dir=dataset_dir,
                    image_id=image_id,
                    cloud_prob_threshold=50  # Ajuste conforme necessidade
                )
                print(f"       [OK] Imagem salva em: {image_file}")

                # Visualizar e pós-processar localmente
                visualize_masks(image_file, remove_components=True, min_size=10)

            except Exception as e:
                print(f"       [ERRO] ao processar {image_id}: {e}")
                continue

if __name__ == "__main__":
    main()
