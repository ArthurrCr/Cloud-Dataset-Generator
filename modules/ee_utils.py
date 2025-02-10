import ee

def initialize_ee(project_id=None):
    """
    Inicializa a conta do Google Earth Engine.
    Assume que a autenticação já foi feita via CLI.
    """
    try:
        ee.Initialize(project=project_id)
        print("Earth Engine inicializado com sucesso!")
    except ee.EEException as e:
        print(f"Erro ao inicializar o Earth Engine: {e}")
        print("Tentando autenticar novamente...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("Autenticação e inicialização concluídas!")

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

# Função para obter as AOIs desenhadas
def get_drawn_aois(map_object):
    """
    Extrai todas as AOIs desenhadas no mapa e as converte para objetos ee.Geometry.
    
    Args:
        map_object (geemap.Map): O objeto do mapa interativo.
    
    Returns:
        list: Uma lista de objetos ee.Geometry.
    """
    drawn_features = map_object.draw_features
    aois = []
    
    if drawn_features:
        for idx, feature in enumerate(drawn_features):
            geom = feature.geometry()
            geojson = geom.getInfo()
            print(f'GeoJSON da AOI {idx + 1}:', geojson)  # Para depuração
            
            try:
                # Utilize ee.Geometry diretamente
                ee_geom = ee.Geometry(geojson)
                aois.append(ee_geom)
                print(f'Geometria AOI {idx + 1} convertida com sucesso.')
            except Exception as e:
                print(f"Falha na conversão da AOI {idx + 1}:", e)
    else:
        print('Nenhuma AOI foi desenhada no mapa.')
    
    return aois

def make_grid(aoi, tile_size_meters):
    """
    Gera uma grade (FeatureCollection) de quadrados do tamanho especificado em metros,
    sobre a AOI fornecida. O retorno são feições (Features) que podem ser usadas para
    exportar porções (tiles) da sua imagem Sentinel-2.

    Parâmetros:
        aoi (ee.Geometry): A área de interesse.
        tile_size_meters (float): Tamanho do lado de cada tile, em metros.
    
    Retorna:
        ee.FeatureCollection: Coleção de polígonos (tiles) no formato de retângulos.
    """

    # 1) Garante que seja um polígono simples (retângulo da AOI) ou gera bounding box da AOI
    aoi_bounds = aoi.bounds()

    # 2) Transforma para EPSG:3857 (projeção métrica)
    #    - O parâmetro maxError=1 (ou 10) define tolerância de reprojeção
    aoi_projected = aoi_bounds.transform(ee.Projection('EPSG:3857'), 1)
    
    # 3) Extrai as coordenadas [ [xMin, yMin], [xMin, yMax], [xMax, yMax], [xMax, yMin], xMin, yMin de novo ]
    #    Aqui, region_coords é uma lista de listas (o polígono transformado).
    region_coords = ee.List(aoi_projected.coordinates().get(0))
    
    # Funções auxiliares para converter e encontrar min/max
    def get_xy(idx):
        return ee.List(region_coords.get(idx))
    
    def get_x(idx):
        return ee.Number(get_xy(idx).get(0))
    
    def get_y(idx):
        return ee.Number(get_xy(idx).get(1))

    # Identifica min_x, max_x, min_y, max_y
    xs = [get_x(0), get_x(1), get_x(2), get_x(3)]
    ys = [get_y(0), get_y(1), get_y(2), get_y(3)]

    x_min = ee.Number(xs[0]).min(xs[1]).min(xs[2]).min(xs[3])
    x_max = ee.Number(xs[0]).max(xs[1]).max(xs[2]).max(xs[3])
    y_min = ee.Number(ys[0]).min(ys[1]).min(ys[2]).min(ys[3])
    y_max = ee.Number(ys[0]).max(ys[1]).max(ys[2]).max(ys[3])

    # 4) Calcula quantos tiles cabem em X e Y
    x_tiles = x_max.subtract(x_min).divide(tile_size_meters).ceil()
    y_tiles = y_max.subtract(y_min).divide(tile_size_meters).ceil()

    # 5) Gera as sequências [0, 1, 2, ..., x_tiles-1] e [0, 1, 2, ..., y_tiles-1]
    x_indices = ee.List.sequence(0, x_tiles.subtract(1))
    y_indices = ee.List.sequence(0, y_tiles.subtract(1))

    # 6) Função interna para gerar cada tile
    def map_y(y_i):
        """Cria os quadrados para um valor de índice y."""
        y_i = ee.Number(y_i)
        def map_x(x_i):
            x_i = ee.Number(x_i)
            x0 = x_min.add(x_i.multiply(tile_size_meters))
            y0 = y_min.add(y_i.multiply(tile_size_meters))
            x1 = x0.add(tile_size_meters)
            y1 = y0.add(tile_size_meters)
            
            # Cria retângulo em EPSG:3857
            tile_poly_3857 = ee.Geometry.Rectangle([x0, y0, x1, y1], 
                                                   proj=ee.Projection('EPSG:3857'), 
                                                   geodesic=False)
            # Retorna para EPSG:4326 e converte em Feature
            tile_poly_4326 = tile_poly_3857.transform(ee.Projection('EPSG:4326'), 1)
            return ee.Feature(tile_poly_4326)
        
        return x_indices.map(map_x)

    # 7) Aplica a função sobre todos os índices y e achata (flatten) a lista
    grid = y_indices.map(map_y).flatten()

    # 8) Converte em FeatureCollection
    grid_fc = ee.FeatureCollection(grid)
    
    # 9) (Opcional) Filtrar para manter apenas os tiles que interceptam a AOI original
    #    Caso queira todos os tiles (mesmo os que "sobram"), não use o .filterBounds
    grid_fc = grid_fc.filterBounds(aoi)

    return grid_fc