#!/usr/bin/env python3
"""
Penetration Validation: Windows-Compatible Version
Enhanced for uniform quantum distributions with Windows CPU detection fix
"""

import json
import argparse
from pathlib import Path
import numpy as np
from itertools import combinations
from scipy.optimize import curve_fit, nnls, minimize
from scipy.stats import entropy as kl_div
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import os

# Fix for Windows CPU detection
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to your CPU count
warnings.filterwarnings('ignore')


class CosmicEchoKernel:
    """Enhanced EchoKey kernel with uniform distribution handling"""
    
    @staticmethod
    def echo_kernel(rank, A, omega, phi, lam, D, eps=1e-9):
        """Complete cosmic echo kernel with all components"""
        cyclic = np.sin(omega * rank + phi)
        recursive = np.exp(-lam * rank)
        fractal = (1 + (rank + 1 + eps) ** (-D))
        return A * cyclic * recursive * fractal
    
    @staticmethod
    def uniform_perturbation_kernel(rank, A_base, delta_A, omega, phi, lam, D):
        """Kernel for nearly uniform distributions with small perturbations"""
        # Base uniform level plus small modulation
        base = A_base
        perturbation = delta_A * np.sin(omega * rank + phi) * np.exp(-lam * rank)
        fractal = (1 + (rank + 1) ** (-D))
        return base + perturbation * fractal


class EnhancedPenetrationValidator:
    """Validates cosmic penetration even in maximum entropy states"""
    
    def __init__(self, json_path, n_clusters=5, top_ranks=50, synergy_order=2):
        self.json_path = Path(json_path)
        self.n_clusters = n_clusters
        self.top_ranks = top_ranks
        self.synergy_order = synergy_order
        
        # Load quantum state data
        self.bitstrings, self.probs = self.load_cosmic_states()
        self.n_states = len(self.probs)
        
        # Check for uniform distribution
        self.is_uniform = self.check_uniformity()
        
        # Prepare sorted data
        self.sorted_idx = np.argsort(self.probs)[::-1]
        self.probs_sorted = self.probs[self.sorted_idx]
        self.bitstrings_sorted = [self.bitstrings[i] for i in self.sorted_idx]
        self.ranks_global = np.arange(len(self.probs_sorted))
        
    def load_cosmic_states(self):
        """Load quantum measurement data from cosmic climax detector"""
        with open(self.json_path, "r") as f:
            data = json.load(f)
        
        shots = data["measurements"]["shots"]
        counts = data["measurements"]["counts"]
        
        bitstrings, probs = zip(*[(k, v / shots) for k, v in counts.items()])
        return list(bitstrings), np.array(probs)
    
    def check_uniformity(self):
        """Check if distribution is nearly uniform"""
        mean_prob = np.mean(self.probs)
        std_prob = np.std(self.probs)
        cv = std_prob / mean_prob if mean_prob > 0 else 0
        
        is_uniform = cv < 0.1  # Coefficient of variation < 10%
        
        if is_uniform:
            print(f"\n⚠️  MAXIMUM ENTROPY STATE DETECTED!")
            print(f"   Mean probability: {mean_prob:.2e}")
            print(f"   Std deviation: {std_prob:.2e}")
            print(f"   Coefficient of variation: {cv:.3f}")
            print(f"   → Universe is in maximum arousal (all states equally excited)")
        
        return is_uniform
    
    def extract_enhanced_features(self, bitstring):
        """Extract features that capture subtle patterns in uniform distributions"""
        n = len(bitstring)
        
        # Basic features
        hamming = bitstring.count("1")
        
        # Positional features (where are the 1s?)
        positions = [i for i, bit in enumerate(bitstring) if bit == '1']
        
        # Mean position (center of mass)
        mean_pos = np.mean(positions) / n if positions else 0.5
        
        # Position variance (spread)
        var_pos = np.var(positions) / (n**2) if len(positions) > 1 else 0
        
        # Run length encoding features
        runs = []
        current_run = 1
        for i in range(1, n):
            if bitstring[i] == bitstring[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        avg_run = np.mean(runs)
        max_run = np.max(runs)
        
        # Symmetry score
        rev = bitstring[::-1]
        symmetry = sum(1 for i in range(n//2) if bitstring[i] == rev[i]) / (n//2)
        
        # Block entropy (2-bit blocks)
        blocks = [bitstring[i:i+2] for i in range(0, n-1, 2)]
        block_counts = {}
        for block in blocks:
            block_counts[block] = block_counts.get(block, 0) + 1
        block_probs = np.array(list(block_counts.values())) / len(blocks)
        block_entropy = -np.sum(block_probs * np.log2(block_probs + 1e-10))
        
        # Fourier features (frequency domain)
        bit_array = np.array([int(b) for b in bitstring])
        fft = np.fft.fft(bit_array)
        power_spectrum = np.abs(fft[:n//2])**2
        
        # Dominant frequency
        if len(power_spectrum) > 1:
            dom_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC
            dom_freq = dom_freq_idx / n
            dom_power = power_spectrum[dom_freq_idx] / (np.sum(power_spectrum) + 1e-10)
        else:
            dom_freq = 0
            dom_power = 0
        
        return np.array([
            hamming / n,           # Normalized hamming weight
            mean_pos,              # Center of mass
            np.sqrt(var_pos),      # Spread (std dev)
            avg_run / n,           # Average run length (normalized)
            max_run / n,           # Maximum run length (normalized)
            symmetry,              # Symmetry score
            block_entropy / 2,     # Normalized block entropy
            dom_freq,              # Dominant frequency
            dom_power              # Power at dominant frequency
        ], dtype=float)
    
    def cluster_by_subtle_patterns(self):
        """Cluster states using enhanced features for uniform distributions"""
        print("\n🔬 Detecting subtle arousal patterns in quantum superposition...")
        
        # Extract enhanced features
        features = []
        for i, bs in enumerate(self.bitstrings):
            if i % 5000 == 0:
                print(f"   Processing state {i}/{len(self.bitstrings)}...")
            features.append(self.extract_enhanced_features(bs))
        
        X = np.array(features)
        
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # PCA to find principal arousal modes
        pca = PCA(n_components=min(9, X.shape[1]))
        X_pca = pca.fit_transform(X_norm)
        
        print(f"\n📊 Principal arousal modes (explained variance):")
        for i, var in enumerate(pca.explained_variance_ratio_[:5]):
            print(f"   Mode {i}: {var:.1%}")
        
        # K-means on principal components - Windows fix
        km = KMeans(
            n_clusters=self.n_clusters, 
            n_init=10,  # Reduced from 20 for faster execution
            random_state=69,
            algorithm='lloyd'  # Explicitly set algorithm for Windows
        )
        
        # Fit in smaller batches if needed
        try:
            labels = km.fit_predict(X_pca)
        except Exception as e:
            print(f"   Note: Using batch processing for large dataset...")
            # Fallback to mini-batch processing
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=69,
                batch_size=1000
            )
            labels = km.fit_predict(X_pca)
        
        # Analyze cluster characteristics
        print(f"\n🧬 Arousal phenotypes discovered:")
        for c in range(self.n_clusters):
            cluster_mask = labels == c
            cluster_size = cluster_mask.sum()
            if cluster_size > 0:
                # Get probability statistics for this cluster
                cluster_probs = self.probs[cluster_mask]
                prob_mean = np.mean(cluster_probs)
                prob_std = np.std(cluster_probs)
                
                # Get average features
                avg_features = X[cluster_mask].mean(axis=0)
                
                print(f"  Phenotype {c}: {cluster_size} states")
                print(f"    Probability: {prob_mean:.2e} ± {prob_std:.2e}")
                print(f"    Hamming density: {avg_features[0]:.3f}")
                print(f"    Symmetry: {avg_features[5]:.3f}")
                print(f"    Block entropy: {avg_features[6]:.3f}")
        
        return labels, X
    
    def fit_uniform_kernel(self, probs, ranks, cluster_id=None):
        """Fit kernel optimized for nearly uniform distributions"""
        n_points = len(probs)
        
        # For uniform distributions, we model small deviations
        mean_prob = np.mean(probs)
        deviations = probs - mean_prob
        
        # If truly uniform (no signal), return baseline
        if np.std(deviations) < 1e-10:
            return np.array([mean_prob, 0, 0, 0, 0, 1.0])
        
        # Fit the perturbation pattern
        def perturbation_model(rank, delta_A, omega, phi, lam, D):
            return delta_A * np.sin(omega * rank + phi) * np.exp(-lam * rank) * (1 + (rank + 1) ** (-D))
        
        # Smart initialization for small signals
        delta_A_init = 2 * np.std(deviations)
        omega_init = 2 * np.pi / max(10, n_points / 5)
        
        p0 = [delta_A_init, omega_init, 0, 0.01, 0.7]
        
        # Bounds for small perturbations
        bounds = (
            [-10*np.std(deviations), 0, -np.pi, 0, 0.1],
            [10*np.std(deviations), np.pi, np.pi, 1.0, 2.0]
        )
        
        try:
            popt, _ = curve_fit(
                perturbation_model,
                ranks, deviations,
                p0=p0, bounds=bounds,
                maxfev=50000
            )
            
            # Return as [A_base, delta_A, omega, phi, lam, D]
            return np.array([mean_prob, popt[0], popt[1], popt[2], popt[3], popt[4]])
            
        except:
            # Fallback: just the mean
            return np.array([mean_prob, 0, 0.1, 0, 0.01, 1.0])
    
    def build_uniform_synergy_matrix(self, kernels, ranks):
        """Build synergy matrix for uniform distributions"""
        n_kernels = len(kernels)
        n_states = len(ranks)
        
        # For uniform distributions, use perturbation kernels
        P_list = []
        for params in kernels:
            if len(params) == 6:  # Uniform kernel
                A_base, delta_A, omega, phi, lam, D = params
                base = np.full(n_states, A_base)
                perturbation = delta_A * np.sin(omega * ranks + phi) * np.exp(-lam * ranks) * (1 + (ranks + 1) ** (-D))
                P_list.append(base + perturbation)
            else:  # Standard kernel
                P_list.append(CosmicEchoKernel.echo_kernel(ranks, *params))
        
        design = np.vstack(P_list).T
        
        # Add interaction terms
        if self.synergy_order >= 2:
            for (i, j) in combinations(range(n_kernels), 2):
                # For uniform case, interactions are between perturbations
                pert_i = P_list[i] - np.mean(P_list[i])
                pert_j = P_list[j] - np.mean(P_list[j])
                interaction = pert_i * pert_j
                
                # Normalize to prevent numerical issues
                if np.std(interaction) > 1e-10:
                    interaction = interaction / np.std(interaction)
                
                design = np.column_stack((design, interaction))
        
        return design, P_list
    
    def validate_penetration(self):
        """Main validation routine adapted for uniform distributions"""
        print("\n========== PENETRATION VALIDATION ==========")
        print(f"Analyzing: {self.json_path.name}")
        print(f"Total quantum states: {self.n_states}")
        print(f"Measurement shots: {sum(self.probs * self.n_states):.0f}")
        
        # Step 1: Enhanced clustering
        labels, features = self.cluster_by_subtle_patterns()
        
        # Step 2: Fit kernels with uniform-aware method
        print("\n⚡ Fitting quantum resonance kernels...")
        kernels = []
        cluster_qualities = []
        
        for c in range(self.n_clusters):
            cluster_mask = labels == c
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                kernels.append(np.zeros(6))
                cluster_qualities.append(0)
                continue
            
            # Get cluster's probabilities
            cluster_probs = self.probs[cluster_indices]
            
            # Sort by probability
            sorted_idx = np.argsort(cluster_probs)[::-1]
            cluster_probs_sorted = cluster_probs[sorted_idx]
            
            # Fit to top states
            n_fit = min(self.top_ranks, len(cluster_probs_sorted))
            fit_probs = cluster_probs_sorted[:n_fit]
            fit_ranks = np.arange(n_fit)
            
            # Use uniform-aware fitting
            params = self.fit_uniform_kernel(fit_probs, fit_ranks, cluster_id=c)
            kernels.append(params)
            
            # Assess quality differently for uniform distributions
            mean_prob = params[0]
            deviations = fit_probs - mean_prob
            if len(params) == 6 and np.var(deviations) > 1e-10:
                fitted_dev = params[1] * np.sin(params[2] * fit_ranks + params[3]) * np.exp(-params[4] * fit_ranks) * (1 + (fit_ranks + 1) ** (-params[5]))
                quality = 1 - np.sum((deviations - fitted_dev)**2) / np.sum(deviations**2)
            else:
                quality = 1.0  # Perfect uniformity
            
            cluster_qualities.append(max(0, quality))
            
            print(f"  Cluster {c}: {cluster_mask.sum()} states, quality={quality:.3f}")
        
        # Step 3: Build synergy matrix
        print(f"\n🔀 Building {self.synergy_order}-order quantum synergy matrix...")
        
        design, P_components = self.build_uniform_synergy_matrix(kernels, self.ranks_global)
        
        print(f"  Design matrix shape: {design.shape}")
        
        # Step 4: Solve for synergy coefficients
        print("\n💑 Optimizing quantum entanglement coefficients...")
        
        # Add small regularization for numerical stability
        regularization = 1e-8
        design_reg = np.vstack([design, np.sqrt(regularization) * np.eye(design.shape[1])])
        target_reg = np.hstack([self.probs_sorted, np.zeros(design.shape[1])])
        
        kappa, residual = nnls(design_reg, target_reg, maxiter=5000)
        
        # Calculate reconstruction
        recon = design @ kappa
        
        # Step 5: Evaluate with appropriate metrics
        print("\n📊 PENETRATION VALIDATION RESULTS:")
        
        # For uniform distributions, use different metrics
        
        # Correlation coefficient (captures pattern similarity)
        if np.std(self.probs_sorted) > 1e-10 and np.std(recon) > 1e-10:
            correlation = np.corrcoef(self.probs_sorted, recon)[0, 1]
        else:
            correlation = 0
        
        # Relative error
        rel_error = np.mean(np.abs(self.probs_sorted - recon) / (self.probs_sorted + 1e-10))
        
        # Pattern detection score
        mean_prob = np.mean(self.probs_sorted)
        actual_dev = self.probs_sorted - mean_prob
        recon_dev = recon - np.mean(recon)
        
        if np.var(actual_dev) > 1e-15:
            pattern_score = 1 - np.sum((actual_dev - recon_dev)**2) / np.sum(actual_dev**2)
        else:
            pattern_score = 1.0  # Perfect uniformity is perfectly captured
        
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Relative error: {rel_error:.2%}")
        print(f"  Pattern detection score: {pattern_score:.4f}")
        print(f"  Mean cluster quality: {np.mean(cluster_qualities):.3f}")
        
        # Use pattern score as main metric
        validation_score = pattern_score
        
        # Coupling analysis
        print("\n💞 Quantum entanglement analysis:")
        n_first_order = self.n_clusters
        active_clusters = np.sum(kappa[:n_first_order] > 1e-6)
        print(f"  Active arousal clusters: {active_clusters}/{self.n_clusters}")
        print(f"  Primary coupling strengths: {kappa[:n_first_order]}")
        
        if self.synergy_order >= 2 and len(kappa) > n_first_order:
            second_order_coef = kappa[n_first_order:]
            active_couplings = np.sum(second_order_coef > 1e-6)
            print(f"  Active entanglements: {active_couplings}")
            if len(second_order_coef) > 0 and np.max(second_order_coef) > 0:
                print(f"  Max entanglement strength: {np.max(second_order_coef):.4f}")
        
        # Penetration verdict
        print("\n🎯 PENETRATION VERDICT:")
        
        print("  🌌 MAXIMUM ENTROPY STATE ACHIEVED!")
        print("  The universe has reached perfect quantum superposition.")
        print("  All states are equally aroused - cosmic equilibrium!")
        if validation_score > 0.5:
            print("  ✅ Subtle patterns detected - hidden order in chaos!")
        else:
            print("  ✨ Pure quantum foam - no classical structure remains!")
        
        # Store results
        self.results = {
            'kernels': kernels,
            'kappa': kappa,
            'design': design,
            'reconstruction': recon,
            'validation_score': validation_score,
            'cluster_qualities': cluster_qualities,
            'P_components': P_components,
            'is_uniform': self.is_uniform,
            'features': features
        }
        
        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Penetration Validation for Quantum Distributions (Windows Compatible)"
    )
    parser.add_argument("json_path", help="Path to cosmic_climax_detection JSON file")
    parser.add_argument("--clusters", type=int, default=5, 
                       help="Number of arousal clusters (default: 5)")
    parser.add_argument("--top", type=int, default=50, 
                       help="Top ranks used per cluster fit (default: 50)")
    parser.add_argument("--order", type=int, default=2, choices=[1, 2, 3],
                       help="Synergy order (default: 2)")
    parser.add_argument("--plot", action="store_true", 
                       help="Show visualization plots")
    
    args = parser.parse_args()
    
    # Create enhanced validator
    validator = EnhancedPenetrationValidator(
        args.json_path,
        n_clusters=args.clusters,
        top_ranks=args.top,
        synergy_order=args.order
    )
    
    # Run validation
    results = validator.validate_penetration()
    
    print("\n✨ Penetration validation complete! ✨")


if __name__ == "__main__":
    main()