import ee

def authenticate_ee():
    """
    Autentica e inicializa a conta do Google Earth Engine.
    """
    ee.Authenticate()
    ee.Initialize()

def merge_s2_and_cloud_prob(s2_sr_col, s2_cp_col):
    """
    Faz um 'join' entre a coleção Sentinel-2 SR e a coleção de cloud probability,
    de modo que cada imagem S2 SR tenha a banda "probability" correspondente.
    Retorna uma coleção de imagens combinadas (com bandas de reflectância e banda 'probability').
    """
    inner_join = ee.Join.inner()
    filter_time_eq = ee.Filter.equals(
        leftField='system:index',
        rightField='system:index'
    )
    joined_collection = inner_join.apply(s2_sr_col, s2_cp_col, filter_time_eq)

    def merge_bands(feature):
        primary = ee.Image(feature.get('primary'))
        secondary = ee.Image(feature.get('secondary'))
        combined = primary.addBands(secondary.select('probability'))
        return combined

    return joined_collection.map(merge_bands)
