import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from modules.mask_processing import remove_small_components

def visualize_masks(image_file, mask_file=None, remove_components=True, min_size=8):
    """
    Visualiza a imagem RGB e as máscaras de nuvem e sombra.

    Existem dois cenários:
      1) Arquivos separados:
         - image_file: TIFF contendo as bandas originais (usaremos B2, B3 e B4 para RGB).
         - mask_file: TIFF pseudo-RGB contendo as máscaras:
               Se o arquivo tiver 3 ou mais bandas:
                 * Banda 1: máscara de nuvem
                 * Banda 3: máscara de sombra
               Se tiver 2 bandas:
                 * Banda 1: máscara de nuvem
                 * Banda 2: máscara de sombra
      2) Arquivo único:
         Se mask_file não for informado, assume-se que image_file contém, na ordem, 
         B2, B3, B4, B8, SCL, probability, cloud_mask, shadow_mask.
    
    Parâmetros:
      image_file: str
          Caminho para o arquivo da imagem (com bandas originais).
      mask_file: str ou None, opcional
          Caminho para o arquivo das máscaras. Se None, as máscaras serão extraídas do image_file.
      remove_components: bool, opcional
          Se True, remove componentes pequenos (default=True).
      min_size: int, opcional
          Tamanho mínimo para manter os componentes (default=8).
    """
    # --- Extração do RGB a partir do arquivo de imagem ---
    with rasterio.open(image_file) as src:
        # Para a imagem exportada pelo Earth Engine, a seleção foi:
        # ['B2', 'B3', 'B4', 'B8', 'SCL', 'probability']
        # Assim, assumimos que:
        # Banda 1: B2 (Blue), Banda 2: B3 (Green), Banda 3: B4 (Red)
        b2 = src.read(1).astype(np.float32)
        b3 = src.read(2).astype(np.float32)
        b4 = src.read(3).astype(np.float32)
        rgb = np.stack([b4, b3, b2], axis=-1)
        max_val = np.max(rgb)
        if max_val > 0:
            rgb /= max_val

    # --- Extração das máscaras ---
    if mask_file:
        with rasterio.open(mask_file) as src:
            band_count = src.count
            if band_count >= 3:
                # Se houver 3 ou mais bandas, usamos banda 1 (nuvem) e banda 3 (sombra)
                cloud_mask = src.read(1)
                shadow_mask = src.read(3)
            elif band_count == 2:
                # Se houver apenas 2 bandas, usamos banda 1 para nuvem e banda 2 para sombra
                cloud_mask = src.read(1)
                shadow_mask = src.read(2)
            else:
                raise ValueError("O arquivo de máscara deve ter pelo menos 2 bandas.")
    else:
        # Cenário de arquivo único: assume que as máscaras estão nas bandas 7 e 8
        with rasterio.open(image_file) as src:
            cloud_mask = src.read(7)
            shadow_mask = src.read(8)

    # Converte as máscaras para valores binários (0 ou 1)
    cloud_mask = (cloud_mask > 0).astype(np.uint8)
    shadow_mask = (shadow_mask > 0).astype(np.uint8)

    # Remoção opcional de componentes pequenos
    if remove_components:
        cloud_mask = remove_small_components(cloud_mask, min_size=min_size)
        shadow_mask = remove_small_components(shadow_mask, min_size=min_size)

    # Aplica operação morfológica (abertura) para suavização
    kernel = np.ones((3, 3), np.uint8)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

    # Exibe as imagens
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (B4, B3, B2)")
    axes[0].axis('off')

    axes[1].imshow(cloud_mask, cmap='Reds')
    axes[1].set_title("Máscara de Nuvem")
    axes[1].axis('off')

    axes[2].imshow(shadow_mask, cmap='Blues')
    axes[2].set_title("Máscara de Sombra")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
