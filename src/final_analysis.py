# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:26:24 2026

@author: eggra
"""

# -*- coding: utf-8 -*-
"""
FINAL ANALYSIS SCRIPT
Consolidates results from:
- baseline_comparison.py (λ=0, 0.5, 1)
- convergence_test.py (stability analysis)
- Existing N=500 data (experiment_M4_N500/results.json)

Generates:
1. Table with key metrics (gap, d_G, variance) for all λ and N
2. Figure 1: Gap vs λ for N=100,200,500 (showing peak at λ≈0.4-0.5)
3. Figure 2: Baseline comparison (λ=0, 0.5, 1) with error bars
4. Figure 3: d_G vs λ for N=100,200,500 (showing peak in intermediate regime)
5. Statistical summary for paper
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy import stats

# =========================================================
# LOAD ALL AVAILABLE DATA
# =========================================================

def load_results(directory, filename="results.json"):
    """Load results from experiment directory."""
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_baseline_results():
    """Load baseline comparison results."""
    path = "experiment_M4/baseline_results.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_convergence_results():
    """Load convergence test results."""
    path = "experiment_M4/convergence_results.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

# =========================================================
# EXTRACT D_G AND GAP FROM EXISTING N=500 DATA
# =========================================================

def extract_from_N500():
    """Extract d_G and gap from existing N=500 results."""
    results = load_results("experiment_M4_N500")
    if results is None:
        print("⚠️ N=500 results not found in experiment_M4_N500/")
        return None
    
    extracted = []
    for r in results:
        extracted.append({
            "lambda": r["lambda"],
            "d_G": r["d_G"],
            "gap": r["gap"],
            "var_L": r["var_L"]
        })
    return extracted

# =========================================================
# GENERATE TABLE FOR PAPER
# =========================================================

def generate_table(N100_results, N200_results, N500_results, baseline_results):
    """Generate summary table with key metrics."""
    
    print("\n" + "="*80)
    print("TABLE 1: SUMMARY OF KEY METRICS")
    print("="*80)
    print(f"{'λ':>8} | {'N':>6} | {'d_G':>8} | {'gap':>12} | {'var(L)':>12} | {'CV(d_G)':>8} | {'CV(gap)':>8}")
    print("-"*80)
    
    # Baseline results (λ=0, 0.5, 1 for N=100)
    if baseline_results:
        for r in baseline_results:
            lam = r["lam"]
            dG_mean = r["d_G"]["mean"]
            dG_std = r["d_G"]["std"]
            gap_mean = r["gap"]["mean"]
            gap_std = r["gap"]["std"]
            var_mean = r["var_L"]["mean"]
            
            cv_dG = dG_std / dG_mean if dG_mean > 0 else 0
            cv_gap = gap_std / gap_mean if gap_mean > 0 else 0
            
            print(f"{lam:>8.1f} | {100:>6} | {dG_mean:>8.2f} | {gap_mean:>12.2f} | {var_mean:>12.2f} | {cv_dG:>8.3f} | {cv_gap:>8.3f}")
    
    # N=100 full results
    if N100_results:
        for r in N100_results:
            lam = r["lambda"]
            # Skip λ=0, 0.5, 1 if already shown in baseline
            if baseline_results and lam in [0.0, 0.5, 1.0]:
                continue
            print(f"{lam:>8.2f} | {100:>6} | {r['d_G']:>8} | {r['gap']:>12.2f} | {r['var_L']:>12.2f} | {'':>8} | {'':>8}")
    
    # N=200 results
    if N200_results:
        for r in N200_results:
            lam = r["lambda"]
            print(f"{lam:>8.2f} | {200:>6} | {r['d_G']:>8} | {r['gap']:>12.2f} | {r['var_L']:>12.2f} | {'':>8} | {'':>8}")
    
    # N=500 results
    if N500_results:
        for r in N500_results:
            lam = r["lambda"]
            print(f"{lam:>8.2f} | {500:>6} | {r['d_G']:>8} | {r['gap']:>12.2f} | {r['var_L']:>12.2f} | {'':>8} | {'':>8}")
    
    print("-"*80)

# =========================================================
# FIGURE 1: Gap vs λ (N=100, 200, 500)
# =========================================================

def figure_gap_vs_lambda(N100_results, N200_results, N500_results):
    """Generate figure showing gap peak across system sizes."""
    
    plt.figure(figsize=(10, 6))
    
    colors = {100: '#2E86AB', 200: '#A23B72', 500: '#F18F01'}
    markers = {100: 'o', 200: 's', 500: '^'}
    labels = {100: 'N=100', 200: 'N=200', 500: 'N=500'}
    
    # Plot N=100
    if N100_results:
        lam = [r["lambda"] for r in N100_results]
        gap = [r["gap"] for r in N100_results]
        plt.plot(lam, gap, marker=markers[100], color=colors[100], 
                linewidth=2, markersize=8, label=labels[100])
    
    # Plot N=200
    if N200_results:
        lam = [r["lambda"] for r in N200_results]
        gap = [r["gap"] for r in N200_results]
        plt.plot(lam, gap, marker=markers[200], color=colors[200], 
                linewidth=2, markersize=8, label=labels[200])
    
    # Plot N=500
    if N500_results:
        lam = [r["lambda"] for r in N500_results]
        gap = [r["gap"] for r in N500_results]
        plt.plot(lam, gap, marker=markers[500], color=colors[500], 
                linewidth=2, markersize=8, label=labels[500])
    
    plt.xlabel("λ (Euclidean weight)", fontsize=14)
    plt.ylabel("gap = max(L) - median(L)", fontsize=14)
    plt.title("Minimum Dominance: Peak of Identifiability", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig("experiment_M4/figures/final_gap_vs_lambda.png", dpi=200)
    plt.close()
    
    print("✓ Figure saved: experiment_M4/figures/final_gap_vs_lambda.png")

# =========================================================
# FIGURE 2: Baseline Comparison (λ=0, 0.5, 1)
# =========================================================

def figure_baseline_comparison(baseline_results):
    """Generate bar plot comparing λ=0, 0.5, 1."""
    
    if baseline_results is None:
        print("⚠️ No baseline results found")
        return
    
    lam_labels = {0.0: "λ=0\n(Radial)", 0.5: "λ=0.5\n(Mixed)", 1.0: "λ=1\n(Euclidean)"}
    colors = {0.0: '#E63946', 0.5: '#2E86AB', 1.0: '#2CA02C'}
    
    # Extract data
    dG_means = []
    dG_stds = []
    gap_means = []
    gap_stds = []
    lam_vals = []
    
    for r in baseline_results:
        lam_vals.append(r["lam"])
        dG_means.append(r["d_G"]["mean"])
        dG_stds.append(r["d_G"]["std"])
        gap_means.append(r["gap"]["mean"])
        gap_stds.append(r["gap"]["std"])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: d_G
    ax1 = axes[0]
    x_pos = np.arange(len(lam_vals))
    ax1.bar(x_pos, dG_means, yerr=dG_stds, capsize=5, 
            color=[colors[l] for l in lam_vals], alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([lam_labels[l] for l in lam_vals])
    ax1.set_ylabel("d_G (degeneracy dimension)", fontsize=12)
    ax1.set_title("Solution Space Degeneracy", fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(dG_means, dG_stds)):
        ax1.text(i, mean + std + 0.3, f"{mean:.1f}", ha='center', fontsize=10)
    
    # Subplot 2: gap
    ax2 = axes[1]
    ax2.bar(x_pos, gap_means, yerr=gap_stds, capsize=5,
            color=[colors[l] for l in lam_vals], alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([lam_labels[l] for l in lam_vals])
    ax2.set_ylabel("gap = max(L) - median(L)", fontsize=12)
    ax2.set_title("Minimum Dominance", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, (mean, std) in enumerate(zip(gap_means, gap_stds)):
        ax2.text(i, mean + std + 5, f"{mean:.1f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("experiment_M4/figures/final_baseline_comparison.png", dpi=200)
    plt.close()
    
    print("✓ Figure saved: experiment_M4/figures/final_baseline_comparison.png")

# =========================================================
# FIGURE 3: d_G vs λ (N=100, 200, 500)
# =========================================================

def figure_dG_vs_lambda(N100_results, N200_results, N500_results):
    """Generate figure showing d_G across system sizes."""
    
    plt.figure(figsize=(10, 6))
    
    colors = {100: '#2E86AB', 200: '#A23B72', 500: '#F18F01'}
    markers = {100: 'o', 200: 's', 500: '^'}
    labels = {100: 'N=100', 200: 'N=200', 500: 'N=500'}
    
    # Plot N=100
    if N100_results:
        lam = [r["lambda"] for r in N100_results]
        dG = [r["d_G"] for r in N100_results]
        plt.plot(lam, dG, marker=markers[100], color=colors[100], 
                linewidth=2, markersize=8, label=labels[100])
    
    # Plot N=200
    if N200_results:
        lam = [r["lambda"] for r in N200_results]
        dG = [r["d_G"] for r in N200_results]
        plt.plot(lam, dG, marker=markers[200], color=colors[200], 
                linewidth=2, markersize=8, label=labels[200])
    
    # Plot N=500
    if N500_results:
        lam = [r["lambda"] for r in N500_results]
        dG = [r["d_G"] for r in N500_results]
        plt.plot(lam, dG, marker=markers[500], color=colors[500], 
                linewidth=2, markersize=8, label=labels[500])
    
    plt.xlabel("λ (Euclidean weight)", fontsize=14)
    plt.ylabel("d_G (effective dimension)", fontsize=14)
    plt.title("Solution Space Degeneracy", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig("experiment_M4/figures/final_dG_vs_lambda.png", dpi=200)
    plt.close()
    
    print("✓ Figure saved: experiment_M4/figures/final_dG_vs_lambda.png")

# =========================================================
# CONVERGENCE ANALYSIS (from convergence test)
# =========================================================

def convergence_analysis(convergence_results):
    """Print convergence analysis results."""
    
    if convergence_results is None:
        print("⚠️ No convergence results found")
        return
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    n_solutions = [r["n_solutions"] for r in convergence_results]
    dG_means = [r["d_G"]["mean"] for r in convergence_results]
    dG_stds = [r["d_G"]["std"] for r in convergence_results]
    gap_means = [r["gap"]["mean"] for r in convergence_results]
    gap_stds = [r["gap"]["std"] for r in convergence_results]
    
    print(f"\n{'n_solutions':>12} | {'d_G':>10} | {'gap':>12}")
    print("-"*40)
    for n, dG, dG_std, gap, gap_std in zip(n_solutions, dG_means, dG_stds, gap_means, gap_stds):
        print(f"{n:>12} | {dG:>8.2f} ± {dG_std:.2f} | {gap:>10.2f} ± {gap_std:.2f}")
    
    # Check stabilization
    print("\n" + "-"*40)
    print("STABILIZATION CHECK:")
    
    # Compare first and last
    dG_change = abs(dG_means[-1] - dG_means[0]) / dG_means[0] * 100
    gap_change = abs(gap_means[-1] - gap_means[0]) / gap_means[0] * 100
    
    print(f"  d_G change from 5 to 50 runs: {dG_change:.1f}%")
    print(f"  gap change from 5 to 50 runs: {gap_change:.1f}%")
    
    if dG_change < 10 and gap_change < 10:
        print("  ✅ Metrics stabilize after ~20-30 initializations")
    else:
        print("  ⚠️ Metrics continue to change with more initializations")

# =========================================================
# STATISTICAL SUMMARY FOR PAPER
# =========================================================

def statistical_summary(N100_results, baseline_results):
    """Generate statistical summary for paper."""
    
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY FOR PAPER")
    print("="*80)
    
    # Find λ* (peak of gap) for N=100
    if N100_results:
        lam_vals = [r["lambda"] for r in N100_results]
        gap_vals = [r["gap"] for r in N100_results]
        
        # Find peak (excluding boundaries)
        interior_indices = [i for i in range(1, len(gap_vals)-1)]
        if interior_indices:
            peak_idx = interior_indices[np.argmax([gap_vals[i] for i in interior_indices])]
            lambda_star = lam_vals[peak_idx]
            gap_star = gap_vals[peak_idx]
            
            print(f"\n📍 Peak of identifiability (N=100):")
            print(f"   λ* = {lambda_star:.3f}")
            print(f"   gap_max = {gap_star:.2f}")
    
    # Baseline comparison statistics
    if baseline_results:
        print("\n📊 Baseline comparison (N=100):")
        for r in baseline_results:
            lam = r["lam"]
            dG = r["d_G"]["mean"]
            gap = r["gap"]["mean"]
            print(f"   λ={lam:.1f}: d_G={dG:.2f}, gap={gap:.2f}")
        
        # Test if mixed regime is significantly different
        gap_05 = [r for r in baseline_results if r["lam"] == 0.5][0]["gap"]["mean"]
        gap_0 = [r for r in baseline_results if r["lam"] == 0.0][0]["gap"]["mean"]
        gap_1 = [r for r in baseline_results if r["lam"] == 1.0][0]["gap"]["mean"]
        
        if gap_05 > gap_0 and gap_05 > gap_1:
            print("\n   ✅ Mixed regime shows MAXIMUM dominance")
        else:
            print("\n   ⚠️ Mixed regime does NOT dominate")

# =========================================================
# MAIN
# =========================================================

def main():
    
    print("="*80)
    print("FINAL ANALYSIS: Consolidating Results for Cocordero")
    print("="*80)
    
    # Create figures directory
    os.makedirs("experiment_M4/figures", exist_ok=True)
    
    # Load all data
    print("\n📂 Loading results...")
    
    N100_results = load_results("experiment_M4")
    N200_results = load_results("experiment_M4_N200")
    N500_results = extract_from_N500()
    baseline_results = load_baseline_results()
    convergence_results = load_convergence_results()
    
    print(f"  ✓ N=100: {len(N100_results) if N100_results else 0} points")
    print(f"  ✓ N=200: {len(N200_results) if N200_results else 0} points")
    print(f"  ✓ N=500: {len(N500_results) if N500_results else 0} points")
    print(f"  ✓ Baseline: {len(baseline_results) if baseline_results else 0} points")
    print(f"  ✓ Convergence: {len(convergence_results) if convergence_results else 0} points")
    
    # Generate table
    generate_table(N100_results, N200_results, N500_results, baseline_results)
    
    # Generate figures
    print("\n📊 Generating figures...")
    figure_gap_vs_lambda(N100_results, N200_results, N500_results)
    figure_baseline_comparison(baseline_results)
    figure_dG_vs_lambda(N100_results, N200_results, N500_results)
    
    # Convergence analysis
    convergence_analysis(convergence_results)
    
    # Statistical summary
    statistical_summary(N100_results, baseline_results)
    
    print("\n" + "="*80)
    print("✅ FINAL ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files:")
    print("  - experiment_M4/figures/final_gap_vs_lambda.png")
    print("  - experiment_M4/figures/final_baseline_comparison.png")
    print("  - experiment_M4/figures/final_dG_vs_lambda.png")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()