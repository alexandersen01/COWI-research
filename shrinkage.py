from shapely import geometry
import matplotlib.pyplot as plt


def shrink(coords, factor=0.1):
    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)

    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))

    center = geometry.Point(x_center, y_center)
    shrink_dist = center.distance(min_corner) * factor

    my_poly = geometry.Polygon(coords)
    my_poly_shrunk = my_poly.buffer(-shrink_dist)

    x, y = my_poly.exterior.xy
    plt.plot(x, y)
    x, y = my_poly_shrunk.exterior.xy
    plt.plot(x, y)

    plt.axis("equal")
    plt.show()

    return list(zip(x, y))


room_vertices = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]

print(shrink(room_vertices))

# https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates, by user F. Sonntag 10.11.2020, retrieved 21.04.2025
