from math import sqrt


def _normalize(v):
    norm = sqrt(v[0] * v[0] + v[1] * v[1])
    if norm > 0:
        return v[0] / norm, v[1] / norm
    else:
        return 0, 0


def _dot(a, b):
    try:
        return a[0] - b[0], a[1] - b[1]
    except:
        return a.x - b[0], a.y - b[1]


def _edge_direction(p0, p1):
    try:
        return p1[0] - p0[0], p1[1] - p0[1]
    except:
        return p1.x - p0.x, p1.y - p0.y


def _orthogonal(v):
    return v[1], -v[0]


def _vertices_to_edges(vertices):
    return [_edge_direction(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _project(vertices, axis):
    dots = [_dot(vertex, axis) for vertex in vertices]
    return [min(dots), max(dots)]


def _contains(n, range_):
    a = range_[0]
    b = range_[1]
    if b < a:
        a = range_[1]
        b = range_[0]
    return (n >= a) and (n <= b)


def _overlap(a, b):
    return _contains(a[0], b) or _contains(a[1], b) or _contains(b[0], a) or _contains(b[1], a)


def separating_axis_theorem(vertices_a, vertices_b):
    edges_a = _vertices_to_edges(vertices_a)
    edges_b = _vertices_to_edges(vertices_b)

    edges = edges_a + edges_b
    axes = [_normalize(_orthogonal(edge)) for edge in edges]

    for i in range(len(axes)):
        if not _overlap(_project(vertices_a, axes[i]), _project(vertices_b, axes[i])):
            return False
    return True
