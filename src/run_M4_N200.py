# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:15:31 2026

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
Experimento M4 para N=200 (script independiente)
Ejecuta el experimento con N=200 y guarda resultados.
Tiempo estimado: ~60-90 minutos.
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

# Intentar tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# =========================================================
# UTILIDADES
# =========================================================

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))

# =========================================================
# INICIALIZACIÓN HÍBRIDA
# =========================================================

def hybrid_initialization(A, lam, d=2):
    """Hybrid initialization: spectral for λ>0.6, random otherwise."""
    N = A.shape[0]
    if lam > 0.6:
        try:
            from scipy.sparse.csgraph import laplacian
            from scipy.sparse.linalg import eigsh
            
            L = laplacian(A, normed=True)
            eigvals, eigvecs = eigsh(L, k=d+1, which='SM', tol=1e-6)
            idx = np.argsort(eigvals)
            X0 = eigvecs[:, idx[1:d+1]]
            X0 = X0 - np.mean(X0, axis=0)
            X0 = X0 / (np.std(X0, axis=0) + 1e-10)
            return X0
        except:
            pass
    return np.random.randn(N, d) * 0.1

# =========================================================
# GENERADOR
# =========================================================

def generate_graph(seed, N=200, alpha=1.5, lam=0.5, d=2):
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
# GRADIENTE VECTORIZADO
# =========================================================

def compute_gradient_vectorized(X, A, alpha, lam):
    N, d = X.shape
    norms = np.linalg.norm(X, axis=1) + 1e-9
    D = pairwise_distances(X)
    D2 = D**2
    d_rad = np.abs(norms[:, None] - norms[None, :])
    sign_rad = np.sign(norms[:, None] - norms[None, :])
    Z = alpha - lam * D2 - (1 - lam) * d_rad
    P = sigmoid(Z)
    np.fill_diagonal(P, 0)
    G = A - P
    grad = np.zeros_like(X)
    for i in range(N):
        diff = X[i] - X
        term1 = -2 * lam * diff
        term2 = -(1 - lam) * sign_rad[i, :, np.newaxis] * (X[i] / norms[i])
        grad[i] = np.sum(G[i, :, np.newaxis] * (term1 + term2), axis=0)
    return grad

# =========================================================
# OPTIMIZACIÓN CON PARADA TEMPRANA
# =========================================================

def optimize_early_stop(A, alpha, lam, d=2, max_steps=400, lr=0.01, tol=1e-6, patience=50):
    N = A.shape[0]
    X = hybrid_initialization(A, lam, d)
    best_X = X.copy()
    best_L = log_likelihood(X, A, alpha, lam)
    no_improve = 0
    
    for step in range(max_steps):
        grad = compute_gradient_vectorized(X, A, alpha, lam)
        X += lr * grad
        
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
# EXPERIMENTO PRINCIPAL
# =========================================================

def run_experiment_N200(lam_values, N=200, seeds=3, n_solutions=15, max_steps=300):
    """Ejecuta experimento para N=200."""
    results = []
    total_optimizations = len(lam_values) * seeds * n_solutions
    
    if HAS_TQDM:
        pbar = tqdm(total=total_optimizations, desc="Optimizations", unit="opt")
    else:
        print(f"Total optimizations: {total_optimizations}")
        pbar = None
    
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
                if pbar:
                    pbar.update(1)
                else:
                    print(f"    Seed {seed}, solution {_+1}/{n_solutions}")
            
            base = solutions[0]
            for X, L in zip(solutions, likelihoods):
                _, X_aligned, _ = procrustes(base, X)
                all_features.append(embedding_to_features(X_aligned))
                all_likelihoods.append(L)
        
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
    
    if pbar:
        pbar.close()
    return results

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    
    print("="*60)
    print("Experiment M4 for N=200")
    print("="*60)
    
    lam_values = np.linspace(0, 1, 10)
    N = 200
    seeds = 3
    n_solutions = 15
    max_steps = 300
    
    print(f"Parameters:")
    print(f"  N = {N}")
    print(f"  λ values: {lam_values}")
    print(f"  seeds = {seeds}")
    print(f"  n_solutions = {n_solutions}")
    print(f"  max_steps = {max_steps}")
    print(f"  Total optimizations: {len(lam_values) * seeds * n_solutions}")
    print()
    
    # Estimación de tiempo
    print("Estimating time...")
    start_bench = time.time()
    A = generate_graph(0, N=N, lam=0.5)
    _, _ = optimize_early_stop(A, alpha=1.5, lam=0.5, max_steps=100, lr=0.01)
    bench_time = time.time() - start_bench
    total_opt = len(lam_values) * seeds * n_solutions
    estimated = bench_time * total_opt / 100 * 300  # 100 steps benchmark, 300 max
    print(f"  Estimated per optimization: {bench_time:.2f}s")
    print(f"  Total estimated time: {estimated/60:.1f} minutes")
    
    if estimated > 300:
        print(f"\n⚠️  Estimated time is {estimated/60:.1f} minutes.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Ejecutar
    print("\nRunning experiment...")
    start_time = time.time()
    results = run_experiment_N200(lam_values, N=N, seeds=seeds, 
                                  n_solutions=n_solutions, max_steps=max_steps)
    elapsed = time.time() - start_time
    
    # Guardar resultados
    os.makedirs("experiment_M4_N200", exist_ok=True)
    with open("experiment_M4_N200/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Experiment completed!")
    print(f"   Real time: {elapsed/60:.1f} minutes")
    print(f"   Results saved in: experiment_M4_N200/results.json")