MONGO_HOST = '192.168.120.127'
MONGO_PORT = 27017
MONGO_DBNAME = 'tmap'
DOMAIN = {
    'district': {},
    'graph_edge': {},
    'graph_node': {},
    'intersection': {},
    'link': {},
    'osm_edge': {},
    'osm_node': {},
    'osm_way': {},
    'road': {},
    'road1': {},
    'road2': {},
}

RESOURCE_METHODS = ['GET', ]

# Enable reads (GET), edits (PATCH), replacements (PUT) and deletes of
# individual items  (defaults to read-only item access).
ITEM_METHODS = ['GET', ]
ALLOW_UNKNOWN = True
