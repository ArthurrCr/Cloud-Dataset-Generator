import numpy as np
from scipy.ndimage import label, generate_binary_structure

def generate_cloud_shadow_masks_sentinel2(image, cloud_prob_threshold=50, morphology_radius=1,
                                          nir_threshold=0.2, cloud_nir_threshold=0.3):
    """
    Retorna duas bandas binárias (cloud_mask, shadow_mask) combinando:
      - Banda SCL (Scene Classification)
      - Banda 'probability' (0-100) da coleção COPERNICUS/S2_CLOUD_PROBABILITY
      - Detecção adicional de sombra usando a banda B8 (NIR) para identificar áreas escuras,
        excluindo pixels de água (SCL == 6)
      - Refinamento morfológico para suavizar as máscaras e eliminar ruídos
    
    Melhorias na máscara de nuvem:
      - Exige que pixels classificados como nuvens (via SCL ou probability) sejam brilhantes no NIR,
        utilizando o limiar cloud_nir_threshold (padrão=0.3) para evitar confusões com superfícies arenosas.
      - Filtra a máscara baseada em probability para descartar pixels com SCL de solo exposto
        (SCL == 5 ou 7), comuns em áreas desérticas.
    
    Parâmetros:
      image: ee.Image
          Imagem Sentinel-2 com as bandas necessárias (SCL, probability, B8).
      cloud_prob_threshold: int, opcional
          Limiar para a banda 'probability' identificar nuvens (padrão=50).
      morphology_radius: int, opcional
          Raio para as operações morfológicas (focal_min e focal_max) (padrão=1).
      nir_threshold: float, opcional
          Limiar (após normalização) para a detecção de áreas escuras no NIR (padrão=0.2).
      cloud_nir_threshold: float, opcional
          Limiar (após normalização) para definir brilho no NIR para pixels de nuvem (padrão=0.3).
    
    Retorna:
      refined_cloud_mask: ee.Image
          Máscara binária de nuvens refinada.
      refined_shadow_mask: ee.Image
          Máscara binária de sombras refinada.
    """
    # Extrai as bandas necessárias
    scl = image.select('SCL')
    cloud_prob = image.select('probability')
    
    # Normaliza a banda B8 para reflectância (0-1) e define como 'nir'
    nir = image.select('B8').divide(10000)
    
    # Define uma condição de brilho no NIR para nuvens – ajustável conforme necessário.
    bright_cloud = nir.gt(cloud_nir_threshold)
    
    # ------------------------------------------------------------------------------
    # 1) DETECÇÃO DE NUVENS
    #    - Via SCL: valores 8 (CLOUD_MEDIUM_PROBA), 9 (CLOUD_HIGH_PROBA),
    #      10 (THIN_CIRRUS) e 11 (SNOW/ICE) indicam nuvens.
    #    - Requer também que o pixel seja brilhante no NIR para evitar confusão com areia.
    # ------------------------------------------------------------------------------
    cloud_mask_scl = (scl.eq(8)
                       .Or(scl.eq(9))
                       .Or(scl.eq(10))
                       .Or(scl.eq(11))
                      ).And(bright_cloud)
    
    #    - Via cloud probability:
    #      Para reduzir falsos positivos sobre áreas arenosas, excluímos pixels com SCL 5 ou 7 (solo exposto)
    #      e exigimos brilho no NIR.
    cloud_mask_prob = (cloud_prob.gt(cloud_prob_threshold)
                       .And(bright_cloud)
                       .And(scl.neq(5))
                       .And(scl.neq(7))
                      )
    
    # Combina as duas detecções para nuvens
    combined_cloud_mask = cloud_mask_scl.Or(cloud_mask_prob)
    
    # Aplica refinamento morfológico para suavizar a máscara de nuvens
    refined_cloud_mask = combined_cloud_mask.focal_min(morphology_radius) \
                                              .focal_max(morphology_radius) \
                                              .rename("cloud_mask")
    
    # ------------------------------------------------------------------------------
    # 2) DETECÇÃO DE SOMBRAS DE NUVEM
    #    - Inicialmente, a partir do SCL: valor 3 indica CLOUD_SHADOW.
    # ------------------------------------------------------------------------------
    shadow_mask_scl = scl.eq(3)
    
    #    - Detecção adicional baseada em áreas escuras no NIR (B8)
    #      Identifica pixels com reflectância menor que o limiar,
    #      excluindo pixels de água (SCL == 6).
    water_mask = scl.eq(6)
    shadow_mask_dark = nir.lt(nir_threshold).And(water_mask.Not())
    
    # Combina as duas abordagens para sombras
    combined_shadow_mask = shadow_mask_scl.Or(shadow_mask_dark)
    
    # Aplica refinamento morfológico para suavizar a máscara de sombras
    refined_shadow_mask = combined_shadow_mask.focal_min(morphology_radius) \
                                               .focal_max(morphology_radius) \
                                               .rename("shadow_mask")
    
    return refined_cloud_mask, refined_shadow_mask

def remove_small_components(mask_array, min_size=8):
    """
    Remove componentes (blobs) menores que 'min_size' pixels usando scipy.ndimage.label.
    
    Parâmetros:
      mask_array: np.array
          Array binário (0 ou 1) representando a máscara.
      min_size: int, opcional
          Tamanho mínimo dos componentes a serem mantidos (padrão=8 pixels).
    
    Retorna:
      np.array
          Array binário (0 ou 1) sem os componentes pequenos.
    """
    structure = generate_binary_structure(2, 2)
    labeled, ncomponents = label(mask_array, structure=structure)

    output = np.copy(mask_array)
    for comp_id in range(1, ncomponents + 1):
        component_mask = (labeled == comp_id)
        if np.sum(component_mask) < min_size:
            output[component_mask] = 0

    return output
