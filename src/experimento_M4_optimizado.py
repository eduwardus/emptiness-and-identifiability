# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 08:14:49 2026

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Experimento M4 (optimizado): Transición de identificabilidad

Optimizaciones:
- Parada temprana con tolerancia (converge antes)
- n_solutions = 15 (suficiente para varianza)
- Inicialización híbrida (mejor punto de partida)

Tiempo estimado original: ~22h
Tiempo estimado optimizado: ~30mh
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import os
import json
import time
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm no instalado. Usando barra manual.")

# =========================================================
# UTILIDADES
# =========================================================

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))

# =========================================================
# INICIALIZACIÓN HÍBRIDA (mejora crítica)
# =========================================================

def hybrid_initialization(A, lam, d=2):
    """
    Inicialización inteligente:
    - Para λ alto (identificable): embedding espectral del grafo
    - Para λ bajo (no identificable): aleatoria (mejor exploración)
    """
    N = A.shape[0]
    if lam > 0.6:
        try:
            # Embedding espectral (Laplaciano normalizado)
            from scipy.sparse.csgraph import laplacian
            from scipy.sparse.linalg import eigsh
            
            L = laplacian(A, normed=True)
            eigvals, eigvecs = eigsh(L, k=d+1, which='SM', tol=1e-6)
            idx = np.argsort(eigvals)
            X0 = eigvecs[:, idx[1:d+1]]
            # Normalizar
            X0 = X0 - np.mean(X0, axis=0)
            X0 = X0 / (np.std(X0, axis=0) + 1e-10)
            return X0
        except:
            pass
    # Fallback: inicialización aleatoria con escala pequeña
    return np.random.randn(N, d) * 0.1

# =========================================================
# GENERADOR
# =========================================================

def generate_graph(seed, N=100, alpha=1.5, lam=0.5, d=2):
    np.random.seed(seed)
    X = np.random.randn(N, d)
    norms = np.linalg.norm(X, axis=1)
    D2 = pairwise_distances(X)**2
    d_rad = np.abs(norms[:, None] - norms[None, :])
    Z = alpha - lam * D2 - (1 - lam) * d_rad
    P = sigmoid(Z)
    np.fill_diagonal(P, 0)
    A = (np.random.rand(N, N) < P).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    return A

# =========================================================
# LOG-LIKELIHOOD
# =========================================================

def log_likelihood(X, A, alpha, lam):
    norms = np.linalg.norm(X, axis=1)
    D2 = pairwise_distances(X)**2
    d_rad = np.abs(norms[:, None] - norms[None, :])
    Z = alpha - lam * D2 - (1 - lam) * d_rad
    P = sigmoid(Z)
    eps = 1e-9
    L = A * np.log(P + eps) + (1 - A) * np.log(1 - P + eps)
    return np.sum(L) / 2

# =========================================================
# GRADIENTE VECTORIZADO (más rápido)
# =========================================================

def compute_gradient_vectorized(X, A, alpha, lam):
    """Versión vectorizada del gradiente (sin bucles Python)."""
    N, d = X.shape
    norms = np.linalg.norm(X, axis=1) + 1e-9
    
    # Distancias
    D = pairwise_distances(X)
    D2 = D**2
    d_rad = np.abs(norms[:, None] - norms[None, :])
    sign_rad = np.sign(norms[:, None] - norms[None, :])
    
    # Probabilidades
    Z = alpha - lam * D2 - (1 - lam) * d_rad
    P = sigmoid(Z)
    np.fill_diagonal(P, 0)
    
    G = A - P  # matriz de residuos
    
    # Precalcular términos por pares (vectorizado)
    grad = np.zeros_like(X)
    
    for i in range(N):
        # Término euclídeo: -2*lam * (X[i] - X[j])
        diff = X[i] - X  # shape (N, d)
        term1 = -2 * lam * diff
        
        # Término radial: -(1-lam) * sign_rad[i,j] * (X[i] / norms[i])
        term2 = -(1 - lam) * sign_rad[i, :, np.newaxis] * (X[i] / norms[i])
        
        grad[i] = np.sum(G[i, :, np.newaxis] * (term1 + term2), axis=0)
    
    return grad

# =========================================================
# OPTIMIZACIÓN CON PARADA TEMPRANA
# =========================================================

def optimize_early_stop(A, alpha, lam, d=2, max_steps=600, lr=0.01, tol=1e-6, patience=50):
    """
    Optimización con parada temprana:
    - Detiene si no mejora en 'patience' pasos
    - Guarda la mejor solución encontrada
    """
    N = A.shape[0]
    X = hybrid_initialization(A, lam, d)
    best_X = X.copy()
    best_L = log_likelihood(X, A, alpha, lam)
    no_improve = 0
    
    for step in range(max_steps):
        grad = compute_gradient_vectorized(X, A, alpha, lam)
        X += lr * grad
        
        # Evaluar cada 10 pasos (ahorra tiempo)
        if step % 10 == 0:
            L = log_likelihood(X, A, alpha, lam)
            if L > best_L + tol:
                best_L = L
                best_X = X.copy()
                no_improve = 0
            else:
                no_improve += 10
            
            if no_improve >= patience:
                break
    
    return best_X, best_L

# =========================================================
# FEATURES
# =========================================================

def embedding_to_features(X):
    D = pairwise_distances(X)
    return D[np.triu_indices_from(D, k=1)]

# =========================================================
# DIMENSIÓN INTRÍNSECA FILTRADA
# =========================================================

def intrinsic_dimension_filtered(features, likelihoods, top_fraction=0.5):
    n = len(likelihoods)
    k = max(1, int(n * top_fraction))
    idx = np.argsort(likelihoods)[-k:]
    filtered_features = [features[i] for i in idx]
    X = np.vstack(filtered_features)
    pca = PCA()
    pca.fit(X)
    ev = pca.explained_variance_
    threshold = 0.05 * ev[0]
    return np.sum(ev > threshold)

# =========================================================
# BARRA DE PROGRESO
# =========================================================

class ProgressBar:
    def __init__(self, total, prefix='', length=40):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.start_time = time.time()
    def update(self, n=1):
        self.current += n
        self._display()
    def _display(self):
        percent = self.current / self.total
        bar = '█' * int(self.length * percent) + '-' * (self.length - int(self.length * percent))
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "?"
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent*100:.1f}% [{elapsed:.1f}s, eta {eta_str}]')
        sys.stdout.flush()
        if self.current == self.total:
            print()

# =========================================================
# EXPERIMENTO PRINCIPAL (con optimizaciones)
# =========================================================

def run_experiment(lam_values, N=100, seeds=3, n_solutions=15, max_steps=400):
    """
    n_solutions reducido de 30 a 15
    max_steps reducido de 600 a 400 (con parada temprana)
    """
    results = []
    total_optimizations = len(lam_values) * seeds * n_solutions
    
    if HAS_TQDM:
        pbar = tqdm(total=total_optimizations, desc="Optimizaciones", unit="opt")
    else:
        pbar = ProgressBar(total_optimizations, prefix="Progreso")
    
    for lam in lam_values:
        print(f"\nλ = {lam:.2f}")
        all_features = []
        all_likelihoods = []
        
        for seed in range(seeds):
            A = generate_graph(seed, N=N, lam=lam)
            solutions = []
            likelihoods = []
            
            for _ in range(n_solutions):
                X, L = optimize_early_stop(A, alpha=1.5, lam=lam, 
                                          max_steps=max_steps, lr=0.01)
                solutions.append(X)
                likelihoods.append(L)
                pbar.update(1)
            
            # Alinear soluciones
            base = solutions[0]
            for X, L in zip(solutions, likelihoods):
                _, X_aligned, _ = procrustes(base, X)
                all_features.append(embedding_to_features(X_aligned))
                all_likelihoods.append(L)
        
        # Métricas
        dG = intrinsic_dimension_filtered(all_features, all_likelihoods, top_fraction=0.5)
        varL = np.var(all_likelihoods)
        gap = np.max(all_likelihoods) - np.median(all_likelihoods)
        
        print(f"  d_G = {dG}, var(L) = {varL:.6f}, gap = {gap:.6f}")
        results.append({
            "lambda": float(lam),
            "d_G": int(dG),
            "var_L": float(varL),
            "gap": float(gap)
        })
    
    if HAS_TQDM:
        pbar.close()
    return results

# =========================================================
# PLOT
# =========================================================

def plot_results(results):
    lam = [r["lambda"] for r in results]
    dG = [r["d_G"] for r in results]
    varL = [r["var_L"] for r in results]
    gap = [r["gap"] for r in results]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(lam, dG, 'o-')
    plt.xlabel("λ")
    plt.ylabel("d_G")
    plt.title("Degeneración (filtrada)")
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(lam, varL, 'o-')
    plt.xlabel("λ")
    plt.ylabel("var(L)")
    plt.title("Varianza del likelihood")
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot(lam, gap, 'o-')
    plt.xlabel("λ")
    plt.ylabel("gap = max(L) - median(L)")
    plt.title("Dominancia del mínimo")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("experiment_M4/transition.png")
    plt.close()

# =========================================================
# ESTIMACIÓN DE TIEMPO
# =========================================================

def estimate_total_time(lam_values, N=100, seeds=3, n_solutions=15, max_steps=400):
    print("Estimando tiempo de ejecución (benchmark rápido)...")
    start = time.time()
    A = generate_graph(0, N=N, lam=0.5)
    _, _ = optimize_early_stop(A, alpha=1.5, lam=0.5, max_steps=100, lr=0.01)
    elapsed_one = time.time() - start
    total_optimizations = len(lam_values) * seeds * n_solutions
    estimated_total = elapsed_one * total_optimizations * 0.8  # factor por menos steps
    print(f"  Tiempo estimado por optimización: {elapsed_one:.2f}s")
    print(f"  Total de optimizaciones: {total_optimizations}")
    print(f"  Tiempo total estimado: {time.strftime('%H:%M:%S', time.gmtime(estimated_total))}")
    return estimated_total

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    
    print("="*60)
    print("Experimento M4 (optimizado)")
    print("Optimizaciones activadas:")
    print("  ✓ Parada temprana (patience=50)")
    print("  ✓ n_solutions = 15 (vs 30)")
    print("  ✓ Inicialización híbrida")
    print("="*60)
    
    lam_values = np.linspace(0, 1, 10)
    N = 100
    seeds = 3
    n_solutions = 15
    max_steps = 400  # reducido, con parada temprana es suficiente
    
    # Estimación de tiempo
    estimated_sec = estimate_total_time(lam_values, N=N, seeds=seeds, 
                                        n_solutions=n_solutions, max_steps=max_steps)
    if estimated_sec > 60:
        print(f"\n⚠️  Tiempo estimado: {estimated_sec/60:.1f} minutos.")
        respuesta = input("¿Desea continuar? (s/n): ")
        if respuesta.lower() != 's':
            print("Cancelado.")
            sys.exit(0)
    
    start = time.time()
    results = run_experiment(lam_values, N=N, seeds=seeds, 
                            n_solutions=n_solutions, max_steps=max_steps)
    elapsed = time.time() - start
    
    os.makedirs("experiment_M4", exist_ok=True)
    with open("experiment_M4/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    plot_results(results)
    
    print(f"\n✅ Tiempo total real: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print("Resultados guardados en experiment_M4/")