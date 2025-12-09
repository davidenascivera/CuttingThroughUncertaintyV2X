import numpy as np
from numba import njit, types
import matplotlib.pyplot as plt

"""
Somma di Minkowski O(n+m) per poligoni convessi orientati CCW.
Ottimizzata con Numba per massime performance.
"""

# -----------------------
# Kernel e utilità Numba ottimizzati
# -----------------------

@njit('int64(float64[:,:])', cache=True, fastmath=True)
def _find_lowest_vertex_index(P):
    """
    Trova l'indice del vertice con y minima (a parità di y, x minima).
    Ottimizzato per performance.
    """
    n = P.shape[0]
    min_idx = 0
    min_y = P[0, 1]
    min_x = P[0, 0]
    
    for i in range(1, n):
        y = P[i, 1]
        x = P[i, 0]
        if y < min_y or (y == min_y and x < min_x):
            min_idx = i
            min_y = y
            min_x = x
    
    return min_idx

@njit('float64[:,:](float64[:,:], int64)', cache=True, fastmath=True)
def _reorder_from_index(P, start_idx):
    """
    Riordina il poligono partendo dall'indice dato.
    Più efficiente della versione precedente.
    """
    n = P.shape[0]
    R = np.empty_like(P)
    
    for i in range(n):
        src_idx = (start_idx + i) % n
        R[i, 0] = P[src_idx, 0]
        R[i, 1] = P[src_idx, 1]
    
    return R

@njit('float64[:,:](float64[:,:], float64[:,:], float64, boolean)', cache=True, fastmath=True)
def _minkowski_convex_kernel_optimized(P, Q, eps_float, float_mode):
    """
    Kernel O(n+m) ottimizzato per la somma di Minkowski.
    Ridotte le allocazioni di memoria e ottimizzati i loop.
    """
    n = P.shape[0]
    m = Q.shape[0]

    # Casi degenerati ottimizzati
    if n == 1:
        if m == 1:
            R = np.empty((1, 2), dtype=np.float64)
            R[0, 0] = P[0, 0] + Q[0, 0]
            R[0, 1] = P[0, 1] + Q[0, 1]
            return R
        else:
            R = np.empty((m, 2), dtype=np.float64)
            px, py = P[0, 0], P[0, 1]
            for j in range(m):
                R[j, 0] = px + Q[j, 0]
                R[j, 1] = py + Q[j, 1]
            return R
    
    if m == 1:
        R = np.empty((n, 2), dtype=np.float64)
        qx, qy = Q[0, 0], Q[0, 1]
        for i in range(n):
            R[i, 0] = P[i, 0] + qx
            R[i, 1] = P[i, 1] + qy
        return R

    # Riordina poligoni dal vertice più basso
    p_start = _find_lowest_vertex_index(P)
    q_start = _find_lowest_vertex_index(Q)
    
    if p_start != 0:
        P = _reorder_from_index(P, p_start)
    if q_start != 0:
        Q = _reorder_from_index(Q, q_start)

    # Pre-alloca risultato (massimo n+m vertici)
    R = np.empty((n + m, 2), dtype=np.float64)
    i = j = k = 0

    # Loop principale ottimizzato
    while i < n or j < m:
        # Aggiungi vertice corrente
        R[k, 0] = P[i % n, 0] + Q[j % m, 0]
        R[k, 1] = P[i % n, 1] + Q[j % m, 1]
        k += 1

        # Calcola direzioni degli spigoli
        next_i = (i + 1) % n
        next_j = (j + 1) % m
        
        # Vettori spigolo (ottimizzato)
        ax = P[next_i, 0] - P[i % n, 0]
        ay = P[next_i, 1] - P[i % n, 1]
        bx = Q[next_j, 0] - Q[j % m, 0]
        by = Q[next_j, 1] - Q[j % m, 1]

        # Cross product
        cross = ax * by - ay * bx

        # Decisione di avanzamento ottimizzata
        if float_mode:
            advance_i = cross >= -eps_float and i < n
            advance_j = cross <= eps_float and j < m
        else:
            advance_i = cross >= 0.0 and i < n
            advance_j = cross <= 0.0 and j < m
        
        if advance_i:
            i += 1
        if advance_j:
            j += 1

    return R[:k]

# -----------------------
# Helper ottimizzati
# -----------------------

@njit('float64(float64[:,:])', cache=True, fastmath=True)
def _signed_area_numba(P):
    """
    Calcolo dell'area orientata con Numba per massima performance.
    """
    n = P.shape[0]
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += P[i, 0] * P[j, 1] - P[j, 0] * P[i, 1]
    
    return area * 0.5

def _ensure_ccw_optimized(P: np.ndarray) -> np.ndarray:
    """
    Versione ottimizzata per forzare orientamento CCW.
    """
    if P.shape[0] < 3:
        return P
    
    area = _signed_area_numba(P.astype(np.float64, copy=False))
    if area < 0.0:
        return np.ascontiguousarray(P[::-1])
    return np.ascontiguousarray(P)

def minkowski_convex_numba_optimized(
    P: np.ndarray,
    Q: np.ndarray,
    ensure_ccw: bool = True,
    eps_float: float = 0.0,
) -> np.ndarray:
    """
    Wrapper ottimizzato con controlli minimi e conversioni efficienti.
    """
    # Validazione minima
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 2 or Q.shape[1] != 2:
        raise ValueError("P e Q devono essere array (N,2) e (M,2).")
    
    # Conversione a float64 ottimizzata
    P_f64 = np.ascontiguousarray(P, dtype=np.float64)
    Q_f64 = np.ascontiguousarray(Q, dtype=np.float64)

    # Forza CCW se richiesto
    if ensure_ccw:
        P_f64 = _ensure_ccw_optimized(P_f64)
        Q_f64 = _ensure_ccw_optimized(Q_f64)

    # Determina modalità float
    float_mode = True  # Sempre true dato che convertiamo a float64

    return _minkowski_convex_kernel_optimized(P_f64, Q_f64, float(eps_float), float_mode)

# -----------------------
# Preprocessing ottimizzato
# -----------------------

@njit('float64[:,:](float64[:,:], float64)', cache=True, fastmath=True)
def _remove_duplicates_numba(A, tol_sq):
    """
    Rimuove duplicati consecutivi usando Numba per performance.
    """
    n = A.shape[0]
    if n <= 1:
        return A.copy()
    
    # Conta vertici unici
    unique_count = 1
    for i in range(1, n):
        dx = A[i, 0] - A[i-1, 0]
        dy = A[i, 1] - A[i-1, 1]
        if dx*dx + dy*dy > tol_sq:
            unique_count += 1
    
    # Crea array risultato
    result = np.empty((unique_count, 2), dtype=np.float64)
    result[0, 0] = A[0, 0]
    result[0, 1] = A[0, 1]
    
    idx = 1
    for i in range(1, n):
        dx = A[i, 0] - A[i-1, 0]
        dy = A[i, 1] - A[i-1, 1]
        if dx*dx + dy*dy > tol_sq:
            result[idx, 0] = A[i, 0]
            result[idx, 1] = A[i, 1]
            idx += 1
    
    return result

def _preprocess_polygon_optimized(A: np.ndarray, coord_tol: float) -> np.ndarray:
    """
    Preprocessing ottimizzato con operazioni Numba.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    
    if A.shape[0] == 0:
        raise ValueError("Poligono vuoto.")
    if A.shape[0] == 1:
        return A

    # Rimuovi chiusura se presente
    tol_sq = coord_tol * coord_tol
    if A.shape[0] > 1:
        dx = A[-1, 0] - A[0, 0]
        dy = A[-1, 1] - A[0, 1]
        if dx*dx + dy*dy <= tol_sq:
            A = A[:-1]

    # Rimuovi duplicati consecutivi con Numba
    if A.shape[0] >= 2:
        A = _remove_duplicates_numba(A, tol_sq)

    return A

@njit('float64(float64[:,:])', cache=True, fastmath=True)
def _max_edge_length_numba(A):
    """
    Calcolo efficiente della lunghezza massima degli spigoli.
    """
    n = A.shape[0]
    if n <= 1:
        return 0.0
    
    max_len_sq = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = A[j, 0] - A[i, 0]
        dy = A[j, 1] - A[i, 1]
        len_sq = dx*dx + dy*dy
        if len_sq > max_len_sq:
            max_len_sq = len_sq
    
    return np.sqrt(max_len_sq)

def minkowski_efficient(P: np.ndarray, Q: np.ndarray, plot_intermediate: bool = False) -> np.ndarray:
    """
    Versione super-ottimizzata della somma di Minkowski O(n+m).
    """
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 2 or Q.shape[1] != 2:
        raise ValueError("P e Q devono essere array (N,2) e (M,2).")

    # Preprocessing ottimizzato
    scale = max(1.0, float(np.max(np.abs(P))), float(np.max(np.abs(Q))))
    coord_tol = 1e-12 * scale

    Pn = _preprocess_polygon_optimized(P, coord_tol)
    Qn = _preprocess_polygon_optimized(Q, coord_tol)

    # Tolleranza adattiva per il cross product
    Lp = _max_edge_length_numba(Pn)
    Lq = _max_edge_length_numba(Qn)
    eps_cross = 1e-12 * max(1.0, Lp * Lq)

    # Calcolo ottimizzato
    R = minkowski_convex_numba_optimized(Pn, Qn, ensure_ccw=True, eps_float=eps_cross)

    if plot_intermediate and R.shape[0] >= 2:
        try:
            plot_minkowski_sum(Pn, Qn, R, "Minkowski Sum (optimized O(n+m))")
        except Exception:
            pass

    return R

# Se vuoi rimpiazzare ovunque minkowski_safe:
# minkowski_safe = minkowski_efficient

# -----------------------
# Plot di supporto
# -----------------------

def plot_minkowski_sum(P, Q, R, title="Minkowski Sum"):
    """
    Plot dei due poligoni P, Q e della loro somma di Minkowski R.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    P_closed = np.vstack([P, P[0]])
    Q_closed = np.vstack([Q, Q[0]])
    R_closed = np.vstack([R, R[0]])

    ax.plot(P_closed[:, 0], P_closed[:, 1], 'b-o', linewidth=2, markersize=6, label='Polygon P')
    ax.fill(P_closed[:, 0], P_closed[:, 1], 'blue', alpha=0.2)

    ax.plot(Q_closed[:, 0], Q_closed[:, 1], 'r-s', linewidth=2, markersize=6, label='Polygon Q')
    ax.fill(Q_closed[:, 0], Q_closed[:, 1], 'red', alpha=0.2)

    ax.plot(R_closed[:, 0], R_closed[:, 1], 'g-^', linewidth=2, markersize=6, label='Minkowski Sum P⊕Q')
    ax.fill(R_closed[:, 0], R_closed[:, 1], 'green', alpha=0.3)

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

# -----------------------
# Esempio d'uso
# -----------------------

if __name__ == "__main__":
    # Primo poligono (convesso, CCW)
    P = np.array([
        [1.90421588e-02, 6.78201658e-03],
        [2.02133624e-02, -1.39790937e-04],
        [2.23605652e-01, -1.54640495e-03],
        [2.10649483e-01, 7.50244918e-02]
    ], dtype=np.float64)

    # Secondo poligono (convesso, contiene quasi-duplicati; NON chiuso qui)
    Q = np.array([
        [21.07868908, -7.56211983],
        [21.01392999, -7.56462129],
        [20.96262732, -6.90357965],
        [21.46324834, -6.82931719],
        [21.53287662, -7.46900064],
        [21.53500730, -7.51022261],
        [21.53250730, -7.51272261]
    ], dtype=np.float64)

    # Somma di Minkowski O(n+m) con normalizzazione e tolleranza adattiva
    R = minkowski_efficient(P, Q, plot_intermediate=False)
    print("Risultato float (k <= n+m): k =", R.shape[0])
    print(R)

    # Plot
    plot_minkowski_sum(P, Q, R, "Minkowski Sum - Custom Polygons")
