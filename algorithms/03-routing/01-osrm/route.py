import osrm

result = osrm.simple_route(
    [21.0566163803209, 42.004088575972], [20.9574645547597, 41.5286973392856],
    output='route', overview="full", geometry='wkt')
result['distance']

result['geometry']
