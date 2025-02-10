import os
import geemap

def create_directories(base_dir=r'data'):
    """
    Cria os diretórios necessários para armazenar as imagens e máscaras.
    As imagens serão salvas em 'data/image' e as máscaras em 'data/mask'.
    """
    base_dir = os.path.abspath(base_dir)
    image_dir = os.path.join(base_dir, 'image')
    mask_dir = os.path.join(base_dir, 'mask')

    try:
        os.makedirs(image_dir, exist_ok=True)
        print(f"Diretório de imagens criado/existente: {image_dir}")
        
        os.makedirs(mask_dir, exist_ok=True)
        print(f"Diretório de máscaras criado/existente: {mask_dir}")
    except Exception as e:
        print(f"Erro ao criar diretórios: {e}")
    
    return image_dir, mask_dir


def download_sentinel_image_and_masks(image, aoi, image_output_dir, mask_output_dir, image_id, cloud_prob_threshold=50):
    """
    Seleciona as bandas de interesse (B2, B3, B4, B8, SCL, probability) da imagem Sentinel,
    gera as máscaras de nuvens e sombras e exporta os resultados em dois arquivos separados:
      - A imagem (com as bandas originais) é salva em image_output_dir.
      - As máscaras (cloud_mask e shadow_mask) são combinadas e salvas em mask_output_dir.
    
    Retorna uma tupla com os caminhos dos arquivos exportados:
      (caminho_da_imagem, caminho_da_máscara)
    """
    # Seleciona as bandas de interesse
    base_bands = ['B2', 'B3', 'B4', 'B8', 'SCL', 'probability']
    selected_image = image.select(base_bands)

    # Importa a função de processamento de máscaras
    from modules.mask_processing import generate_cloud_shadow_masks_sentinel2

    # Gera as máscaras de nuvens e sombras
    cloud_mask, shadow_mask = generate_cloud_shadow_masks_sentinel2(selected_image, cloud_prob_threshold)
    
    # Cria uma imagem contendo as máscaras (bandas combinadas)
    mask_image = cloud_mask.addBands(shadow_mask)

    # Define os nomes e caminhos dos arquivos de saída
    image_filename = os.path.join(image_output_dir, f'sentinel_{image_id}.tif')
    mask_filename = os.path.join(mask_output_dir, f'sentinel_{image_id}_mask.tif')

    image_filename = os.path.normpath(image_filename)
    mask_filename = os.path.normpath(mask_filename)

    abs_image_filename = os.path.abspath(image_filename)
    abs_mask_filename = os.path.abspath(mask_filename)

    print(f"Iniciando exportação da imagem para: {abs_image_filename}")
    print(f"Iniciando exportação das máscaras para: {abs_mask_filename}")

    # Exporta a imagem com as bandas selecionadas
    try:
        geemap.ee_export_image(
            selected_image,  # O objeto é passado como primeiro argumento, sem 'image='
            filename=image_filename,
            scale=10,
            region=aoi,
            file_per_band=False
        )
        print(f"Exportação da imagem concluída: {image_filename}")
    except Exception as e:
        print(f"Erro na exportação da imagem {image_id}: {e}")
        raise e

    # Exporta a imagem com as máscaras
    try:
        geemap.ee_export_image(
            mask_image,  # Também aqui, passando o objeto diretamente
            filename=mask_filename,
            scale=10,
            region=aoi,
            file_per_band=False
        )
        print(f"Exportação das máscaras concluída: {mask_filename}")
    except Exception as e:
        print(f"Erro na exportação das máscaras para a imagem {image_id}: {e}")
        raise e

    return image_filename, mask_filename
