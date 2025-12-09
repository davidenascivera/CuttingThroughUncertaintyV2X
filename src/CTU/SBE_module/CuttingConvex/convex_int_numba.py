import numpy as np
import numba
import os


"""
Convex Polygon Intersection Algorithm using Numba for acceleration.

Input Structure:
- P_in: A flattened array of polygon P vertices as [x1, y1, x2, y2, ..., xn, yn]
  Vertices must be in counter-clockwise order.
- nP: Number of vertices in polygon P.
- Q_in: A flattened array of polygon Q vertices as [x1, y1, x2, y2, ..., xm, ym]
  Vertices must be in counter-clockwise order.
- nQ: Number of vertices in polygon Q.
- max_out: Maximum number of output vertices to allocate space for.

Output:
- An array of intersection points [x1, y1, x2, y2, ..., xk, yk].
- The number of intersection points.

Requirements:
1. Both polygons must be convex.
2. Vertices must be ordered counter-clockwise.
3. The input coordinates should be integer values.
4. The output coordinates will be floating-point values.
"""


USE_NUMBA = os.getenv("USE_NUMBA", "1") == "1"

if USE_NUMBA:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

# Constants
X = 0
Y = 1

@njit
def sub_vec(a, b):
    """Computes the vector difference c = a - b."""
    return np.array([a[X] - b[X], a[Y] - b[Y]], dtype=np.int64)

@njit
def dot(a, b):
    """Returns the dot product of vectors a and b."""
    return a[X] * b[X] + a[Y] * b[Y]

@njit
def area_sign(a, b, c):
    """
    Returns sign of area of triangle (a, b, c).
    Positive if c is to the left of ab, negative if to the right.
    """
    area2 = (b[X] - a[X]) * (c[Y] - a[Y]) - (c[X] - a[X]) * (b[Y] - a[Y])
    if area2 > 0.5:
        return 1
    elif area2 < -0.5:
        return -1
    else:
        return 0

@njit
def between(a, b, c):
    """
    Returns True if point c lies on the closed segment ab.
    Assumes a, b, c are collinear.
    """
    if a[X] != b[X]:
        return ((a[X] <= c[X]) and (c[X] <= b[X])) or ((a[X] >= c[X]) and (c[X] >= b[X]))
    else:
        return ((a[Y] <= c[Y]) and (c[Y] <= b[Y])) or ((a[Y] >= c[Y]) and (c[Y] >= b[Y]))

@njit
def parallel_int(a, b, c, d):
    """Handles the case when segments ab and cd are parallel."""
    if not between(a, b, c):
        return '0', np.zeros(2), np.zeros(2)
    
    # For collinear overlaps, choose endpoints as appropriate.
    if between(a, b, c) and between(a, b, d):
        return 'e', c.astype(np.int64), d.astype(np.int64)
        
    if between(c, d, a) and between(c, d, b):
        return 'e', a.astype(np.int64), b.astype(np.int64)
        
    if between(a, b, c) and between(c, d, b):
        return 'e', c.astype(np.int64), b.astype(np.int64)
        
    if between(a, b, c) and between(c, d, a):
        return 'e', c.astype(np.int64), a.astype(np.int64)
        
    if between(a, b, d) and between(c, d, b):
        return 'e', d.astype(np.int64), b.astype(np.int64)
        
    if between(a, b, d) and between(c, d, a):
        return 'e', d.astype(np.int64), a.astype(np.int64)
        
    return '0', np.zeros(2), np.zeros(2)

@njit
def seg_seg_int(a, b, c, d):
    """
    Finds the point of intersection p between closed segments ab and cd.
    Returns a code:
        '1': Proper intersection (internal to both segments).
        'v': An endpoint of one segment is on the other.
        'e': The segments are collinear and overlapping.
        '0': No intersection.
    The computed intersection point is stored in p.
    """
    denom = a[X] * (d[Y] - c[Y]) + b[X] * (c[Y] - d[Y]) + \
            d[X] * (b[Y] - a[Y]) + c[X] * (a[Y] - b[Y])
    
    if denom == 0.0:
        return parallel_int(a, b, c, d)
    
    num = a[X] * (d[Y] - c[Y]) + c[X] * (a[Y] - d[Y]) + d[X] * (c[Y] - a[Y])
    
    code = '?'
    if (num == 0.0) or (num == denom):
        code = 'v'
    s = num / denom
    
    num = -(a[X] * (c[Y] - b[Y]) + b[X] * (a[Y] - c[Y]) + c[X] * (b[Y] - a[Y]))
    if (num == 0.0) or (num == denom):
        code = 'v'
    t = num / denom
    
    if (0.0 < s) and (s < 1.0) and (0.0 < t) and (t < 1.0):
        code = '1'
    elif (s < 0.0) or (s > 1.0) or (t < 0.0) or (t > 1.0):
        code = '0'
    
    p = np.array([
        a[X] + s * (b[X] - a[X]),
        a[Y] + s * (b[Y] - a[Y])
    ], dtype=np.int64)
    
    q = np.zeros(2, dtype=np.int64)  # Just a placeholder for the parallel case
    
    return code, p, q

@njit
def store_point(out_points, count, max_out, p):
    """Store intersection point in output array."""
    if count < max_out:
        out_points[2 * count] = p[X]
        out_points[2 * count + 1] = p[Y]
        return count + 1
    return count

@njit
def in_out_collect(p, inflag, ahb, bha, out_points, count, max_out):
    """Updates the in/out flag and stores the intersection point."""
    count = store_point(out_points, count, max_out, p)
    if ahb > 0:
        return "Pin", count
    elif bha > 0:
        return "Qin", count
    else:
        return inflag, count

@njit
def advance_collect(a, aa, n, inside, v, out_points, count, max_out):
    """Advances index and, if "inside" is True, collects the point."""
    if inside:
        pv = v.astype(np.int64)
        count = store_point(out_points, count, max_out, pv)
    aa += 1
    return (a + 1) % n, aa, count

@njit
def point_in_convex_polygon(p, poly, n):
    """
    Check if point p is inside the convex polygon poly with n vertices.
    Returns True if inside or on boundary, False otherwise.
    """
    for i in range(n):
        j = (i + 1) % n
        # If p is to the right of any edge, it's outside
        if area_sign(poly[i], poly[j], p) < 0:
            return False
    return True

@njit
def convex_intersect(P_in, nP, Q_in, nQ, max_out):
    """
    Computes the intersection points of two convex polygons.
    Input:
        P_in: array of 2*nP ints (x,y pairs) for polygon P.
        nP:  number of vertices in polygon P.
        Q_in: array of 2*nQ ints for polygon Q.
        nQ:  number of vertices in polygon Q.
        max_out: maximum number of points to store.
    Returns:
        The output points array and number of intersection points.
    """
    # Local polygon copies reshaped to 2D arrays
    P = np.zeros((nP, 2), dtype=np.int64)
    Q = np.zeros((nQ, 2), dtype=np.int64)
    
    # Fill polygon P
    for i in range(nP):
        P[i, X] = P_in[2 * i]
        P[i, Y] = P_in[2 * i + 1]
    
    # Fill polygon Q
    for i in range(nQ):
        Q[i, X] = Q_in[2 * i]
        Q[i, Y] = Q_in[2 * i + 1]
    
    # Output array
    out_points = np.zeros(2 * max_out, dtype=np.int64)
    count = 0
    
    # Check if P is inside Q
    p_inside_q = True
    for i in range(nP):
        if not point_in_convex_polygon(P[i], Q, nQ):
            p_inside_q = False
            break
    
    # If P is inside Q, return P as the intersection
    if p_inside_q:
        for i in range(nP):
            if count < max_out:
                out_points[2 * count] = P[i, X]
                out_points[2 * count + 1] = P[i, Y]
                count += 1
        # Close the polygon by repeating first point
        if count < max_out and count > 0:
            out_points[2 * count] = P[0, X]
            out_points[2 * count + 1] = P[0, Y]
            count += 1
        return out_points, count
    
    # Check if Q is inside P
    q_inside_p = True
    for i in range(nQ):
        if not point_in_convex_polygon(Q[i], P, nP):
            q_inside_p = False
            break
    
    # If Q is inside P, return Q as the intersection
    if q_inside_p:
        for i in range(nQ):
            if count < max_out:
                out_points[2 * count] = Q[i, X]
                out_points[2 * count + 1] = Q[i, Y]
                count += 1
        # Close the polygon by repeating first point
        if count < max_out and count > 0:
            out_points[2 * count] = Q[0, X]
            out_points[2 * count + 1] = Q[0, Y]
            count += 1
        return out_points, count
    
    # Variables for the algorithm
    a, b = 0, 0
    aa, ba = 0, 0
    inflag = "Unknown"
    first_point = True
    p0 = np.zeros(2, dtype=np.int64)
    origin = np.array([0, 0], dtype=np.int64)
    
    while ((aa < nP) or (ba < nQ)) and (aa < 2*nP) and (ba < 2*nQ):
        a1 = (a + nP - 1) % nP
        b1 = (b + nQ - 1) % nQ
        
        A = sub_vec(P[a], P[a1])
        B = sub_vec(Q[b], Q[b1])
        
        cross = area_sign(origin, A, B)
        ahb = area_sign(Q[b1], Q[b], P[a])
        bha = area_sign(P[a1], P[a], Q[b])
        
        code, p, q = seg_seg_int(P[a1], P[a], Q[b1], Q[b])
        
        if code == '1' or code == 'v':
            if inflag == "Unknown" and first_point:
                aa = ba = 0
                first_point = False
                p0[X] = p[X]
                p0[Y] = p[Y]
                count = store_point(out_points, count, max_out, p0)
            
            inflag, count = in_out_collect(p, inflag, ahb, bha, out_points, count, max_out)
        
        # Special case: overlapping segments (oppositely oriented)
        if code == 'e' and dot(A, B) < 0:
            count = store_point(out_points, count, max_out, p)
            count = store_point(out_points, count, max_out, q)
            return out_points, count
        
        # Special case: segments parallel and separated
        if cross == 0 and ahb < 0 and bha < 0:
            # Polygons are disjoint
            return out_points, 0
        
        # Special case: collinear segments
        if cross == 0 and ahb == 0 and bha == 0:
            inside_p = inflag == "Pin"
            inside_q = inflag == "Qin"
            
            if inflag == "Pin":
                b, ba, count = advance_collect(b, ba, nQ, inside_q, Q[b], out_points, count, max_out)
            else:
                a, aa, count = advance_collect(a, aa, nP, inside_p, P[a], out_points, count, max_out)
        
        # Generic advance rules
        elif cross >= 0:
            inside_p = inflag == "Pin"
            inside_q = inflag == "Qin"
            
            if bha > 0:
                a, aa, count = advance_collect(a, aa, nP, inside_p, P[a], out_points, count, max_out)
            else:
                b, ba, count = advance_collect(b, ba, nQ, inside_q, Q[b], out_points, count, max_out)
        
        else:  # cross < 0
            inside_p = inflag == "Pin"
            inside_q = inflag == "Qin"
            
            if ahb > 0:
                b, ba, count = advance_collect(b, ba, nQ, inside_q, Q[b], out_points, count, max_out)
            else:
                a, aa, count = advance_collect(a, aa, nP, inside_p, P[a], out_points, count, max_out)
    
    if not first_point:
        # Close the intersection polygon by repeating the first point
        count = store_point(out_points, count, max_out, p0)
    
    return out_points, count
