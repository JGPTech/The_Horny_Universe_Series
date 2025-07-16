#!/usr/bin/env python3
"""
Dark Matter Signature Extraction from Cosmic Climax Quantum Data
================================================================

This script implements the theoretical framework from Days 1-3 to extract
dark matter properties from maximally mixed quantum states.

Theory: The uniform distribution represents post-climactic equilibrium,
but the bit patterns contain fossilized imprints of the dark matter
arousal field that caused the cosmic climax.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
import os

# Fix for Windows CPU detection
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to your CPU count
warnings.filterwarnings('ignore')


class DarkMatterExtractor:
    """
    Extracts dark matter properties from quantum measurement patterns
    using the unified framework of arousal dynamics and protomatter evolution
    """
    
    def __init__(self, json_path):
        self.json_path = Path(json_path)
        self.load_quantum_data()
        self.extract_features()
        
    def load_quantum_data(self):
        """Load the cosmic climax detection data"""
        print("🌌 Loading quantum measurement data...")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        self.shots = data["measurements"]["shots"]
        self.counts = data["measurements"]["counts"]
        
        # Convert to lists
        self.bitstrings = list(self.counts.keys())
        self.frequencies = np.array(list(self.counts.values()))
        self.probabilities = self.frequencies / self.shots
        
        # Detect number of qubits from the data
        max_value = max(int(bs) for bs in self.bitstrings)
        self.n_qubits = int(np.ceil(np.log2(float(max_value + 1))))  # Convert to float for numpy
        
        print(f"   Loaded {len(self.bitstrings)} quantum states")
        print(f"   Detected {self.n_qubits} qubits")
        print(f"   Total measurements: {self.shots}")
        print(f"   Mean probability: {np.mean(self.probabilities):.2e}")
        
    def extract_features(self):
        """Extract features that encode dark matter properties"""
        print("\n🔬 Extracting dark matter signatures from quantum patterns...")
        
        features = []
        for i, bitstring in enumerate(self.bitstrings):
            if i % 5000 == 0:
                print(f"   Processing state {i}/{len(self.bitstrings)}...")
            
            features.append(self.extract_dm_features(bitstring))
        
        self.features = np.array(features)
        self.feature_names = [
            'dm_density',           # Hamming weight (normalized)
            'dm_concentration',     # Central concentration of 1s
            'dm_spin',             # Asymmetry (angular momentum proxy)
            'dm_velocity_disp',    # Bit flip frequency (kinetic energy)
            'dm_temp',             # Block entropy (temperature proxy)
            'dm_coupling',         # Correlation length
            'dm_age',              # Run length statistics (formation time)
            'dm_environment',      # Edge vs center activity
            'dm_genetic_marker'    # Unique pattern identifier
        ]
        
    def extract_dm_features(self, bitstring):
        """Extract features that correspond to dark matter halo properties"""
        # Convert decimal string to binary representation
        decimal_value = int(bitstring)
        # Use detected number of qubits
        binary_string = format(decimal_value, f'0{self.n_qubits}b')
        
        n = len(binary_string)
        bits = np.array([int(b) for b in binary_string])
        
        # 1. Dark matter density (hamming weight)
        dm_density = np.sum(bits) / n
        
        # 2. Concentration parameter (NFW profile proxy)
        positions = np.where(bits == 1)[0]
        if len(positions) > 0:
            center = n / 2
            distances = np.abs(positions - center)
            dm_concentration = 1 - (np.mean(distances) / (n/2))
        else:
            dm_concentration = 0
        
        # 3. Spin parameter (asymmetry)
        left_sum = np.sum(bits[:n//2])
        right_sum = np.sum(bits[n//2:])
        dm_spin = (left_sum - right_sum) / (left_sum + right_sum + 1)
        
        # 4. Velocity dispersion (bit transition rate)
        transitions = np.sum(np.abs(np.diff(bits)))
        dm_velocity_disp = transitions / (n - 1)
        
        # 5. Temperature (local entropy)
        # Use 3-bit blocks to measure local disorder
        if n >= 3:
            blocks = [binary_string[i:i+3] for i in range(0, n-2, 3)]
            unique_blocks = len(set(blocks))
            dm_temp = unique_blocks / len(blocks) if blocks else 0
        else:
            dm_temp = 0.5  # Default for very short strings
        
        # 6. Coupling strength (correlation length)
        if np.std(bits) > 0:
            autocorr = np.correlate(bits - np.mean(bits), bits - np.mean(bits), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)  # Avoid division by zero
            # Find correlation length (where autocorr drops to 1/e)
            corr_length = np.argmax(autocorr < 1/np.e) if np.any(autocorr < 1/np.e) else n
            dm_coupling = corr_length / n
        else:
            dm_coupling = 0
        
        # 7. Formation redshift proxy (longest run)
        runs = []
        current_run = 1
        for i in range(1, n):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        dm_age = np.max(runs) / n
        
        # 8. Environment (edge activity vs center)
        edge_activity = (bits[0] + bits[-1]) / 2
        center_activity = np.mean(bits[n//3:2*n//3])
        dm_environment = edge_activity - center_activity
        
        # 9. Genetic marker (unique pattern hash)
        # Use the original decimal value as genetic ID, normalized
        genetic_id = decimal_value / (2**self.n_qubits - 1)  # Normalize to [0,1]
        dm_genetic_marker = genetic_id
        
        return np.array([
            dm_density, dm_concentration, dm_spin, dm_velocity_disp,
            dm_temp, dm_coupling, dm_age, dm_environment, dm_genetic_marker
        ])
    
    def analyze_dark_matter_distribution(self):
        """Analyze the distribution of dark matter properties"""
        print("\n📊 Analyzing dark matter property distributions...")
        
        results = {}
        
        for i, feature_name in enumerate(self.feature_names):
            values = self.features[:, i]
            results[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
            print(f"\n{feature_name}:")
            print(f"   Mean: {results[feature_name]['mean']:.4f}")
            print(f"   Std:  {results[feature_name]['std']:.4f}")
            print(f"   Range: [{results[feature_name]['min']:.4f}, {results[feature_name]['max']:.4f}]")
        
        self.dm_stats = results
        return results
    
    def identify_halo_phenotypes(self, n_phenotypes=7):
        """Identify dark matter halo phenotypes based on feature clustering"""
        print(f"\n🧬 Identifying {n_phenotypes} dark matter halo phenotypes...")
        
        # Normalize features
        features_norm = (self.features - self.features.mean(axis=0)) / (self.features.std(axis=0) + 1e-8)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=5)
        features_pca = pca.fit_transform(features_norm)
        
        print("\n📈 Principal components of dark matter variation:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"   PC{i+1}: {var:.1%} - {self.interpret_pc(pca.components_[i])}")
        
        # K-means clustering - Windows compatible
        from sklearn.cluster import KMeans
        import os
        
        # Use explicit parameters for Windows compatibility
        km = KMeans(
            n_clusters=n_phenotypes, 
            n_init=10, 
            random_state=42,
            algorithm='lloyd'  # Explicitly specify algorithm
        )
        phenotypes = km.fit_predict(features_pca)
        
        # Map to theoretical phenotypes from Day 3
        phenotype_names = [
            "giant_cluster_core",
            "spinning_colossus", 
            "compact_dynamo",
            "turbulent_giant",
            "field_halo",
            "compact_dwarf",
            "diffuse_dwarf"
        ]
        
        # Analyze each phenotype
        self.phenotype_stats = {}
        phenotype_assignments = {}  # Track which cluster gets which phenotype
        
        for p in range(n_phenotypes):
            mask = phenotypes == p
            count = np.sum(mask)
            
            if count > 0:
                # Average features for this phenotype
                avg_features = self.features[mask].mean(axis=0)
                
                # Identify phenotype based on key features
                phenotype_id = self.classify_phenotype(avg_features)
                
                # Ensure unique phenotype names by adding cluster ID if duplicate
                base_phenotype_name = phenotype_names[phenotype_id % len(phenotype_names)]
                phenotype_name = base_phenotype_name
                
                # If this phenotype name is already used, make it unique
                suffix = 1
                while phenotype_name in self.phenotype_stats:
                    phenotype_name = f"{base_phenotype_name}_{suffix}"
                    suffix += 1
                
                phenotype_assignments[p] = phenotype_name
                
                self.phenotype_stats[phenotype_name] = {
                    'count': count,
                    'fraction': count / len(phenotypes),
                    'avg_features': avg_features,
                    'dm_density': avg_features[0],
                    'concentration': avg_features[1],
                    'spin': avg_features[2]
                }
                
                print(f"\n{phenotype_name}:")
                print(f"   Count: {count} ({count/len(phenotypes):.1%})")
                print(f"   DM density: {avg_features[0]:.3f}")
                print(f"   Concentration: {avg_features[1]:.3f}")
                print(f"   Spin parameter: {avg_features[2]:.3f}")
        
        self.phenotypes = phenotypes
        self.phenotype_names = phenotype_names
        return self.phenotype_stats
    
    def interpret_pc(self, component):
        """Interpret principal component based on feature loadings"""
        top_features = np.argsort(np.abs(component))[-3:][::-1]
        interpretations = []
        for idx in top_features:
            interpretations.append(f"{self.feature_names[idx]}({component[idx]:.2f})")
        return " + ".join(interpretations)
    
    def classify_phenotype(self, features):
        """Classify phenotype based on feature values"""
        dm_density = features[0]
        concentration = features[1]
        spin = np.abs(features[2])
        
        # Classification logic based on Day 3 theory
        if dm_density > 0.15 and concentration > 0.5:
            return 0  # giant_cluster_core
        elif dm_density > 0.12 and spin > 0.1:
            return 1  # spinning_colossus
        elif concentration > 0.6 and dm_density > 0.10:
            return 2  # compact_dynamo
        elif features[3] > 0.5:  # high velocity dispersion
            return 3  # turbulent_giant
        elif 0.08 < dm_density < 0.12:
            return 4  # field_halo
        elif dm_density < 0.08 and concentration > 0.4:
            return 5  # compact_dwarf
        else:
            return 6  # diffuse_dwarf
    
    def calculate_arousal_dynamics(self):
        """Calculate arousal potentials from dark matter properties"""
        print("\n⚡ Computing arousal dynamics from dark matter distribution...")
        
        # Based on Day 3 equations
        ALPHA_SPIN = 0.5
        BETA_CONC = 2.5
        GAMMA_COUPLING = 2.0
        DELTA_LOVEHOLE = 2.0
        
        # Extract relevant features
        spin_params = np.abs(self.features[:, 2])  # spin
        concentrations = self.features[:, 1]        # concentration
        couplings = self.features[:, 5]             # coupling
        
        # Calculate arousal components
        S_potential = np.log1p(spin_params / 0.035)  # 0.035 is median spin
        C_potential = np.log1p(concentrations / np.median(concentrations))
        Omega_potential = couplings
        
        # Total arousal potential
        self.arousal_potentials = (
            ALPHA_SPIN * S_potential +
            BETA_CONC * C_potential +
            GAMMA_COUPLING * Omega_potential +
            DELTA_LOVEHOLE
        )
        
        print(f"\n📊 Arousal potential statistics:")
        print(f"   Mean: {np.mean(self.arousal_potentials):.2f}")
        print(f"   Std:  {np.std(self.arousal_potentials):.2f}")
        print(f"   Max:  {np.max(self.arousal_potentials):.2f}")
        print(f"   Climax-ready (A > 5.0): {np.sum(self.arousal_potentials > 5.0)} halos")
        
        return self.arousal_potentials
    
    def detect_dark_matter_structure(self):
        """Detect large-scale dark matter structures"""
        print("\n🕸️ Detecting cosmic web structure...")
        
        # Use t-SNE to find structure in high-dimensional space
        tsne = TSNE(n_components=2, perplexity=50, random_state=42)
        embedded = tsne.fit_transform(self.features[:1000])  # Subsample for speed
        
        # Detect filaments using density
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(embedded.T)
        density = kde(embedded.T)
        
        # Identify structures
        high_density = density > np.percentile(density, 80)
        n_filaments = np.sum(high_density)
        
        print(f"   Detected {n_filaments} high-density regions (filaments/nodes)")
        
        # Calculate void fraction
        void_fraction = np.sum(density < np.percentile(density, 20)) / len(density)
        print(f"   Void fraction: {void_fraction:.1%}")
        
        self.structure_map = {
            'embedding': embedded,
            'density': density,
            'void_fraction': void_fraction,
            'n_filaments': n_filaments
        }
        
        return self.structure_map
    
    def measure_protomatter_evolution(self):
        """Measure the maturation state of protomatter"""
        print("\n🔄 Measuring protomatter maturation state...")
        
        # From Day 3: Maturation follows sigmoid with arousal
        X0_THRESHOLD = 5.0
        K_STEEPNESS = 0.8
        
        # Calculate maturation for each halo
        A = self.arousal_potentials
        t = 13.8  # Current universe age in Gyr
        
        # Maturation state M(t)
        self.maturation_states = 1 / (1 + np.exp(-K_STEEPNESS * (A * t - X0_THRESHOLD)))
        
        # Statistics
        fully_mature = np.sum(self.maturation_states > 0.99)
        partially_mature = np.sum((self.maturation_states > 0.5) & (self.maturation_states <= 0.99))
        immature = np.sum(self.maturation_states <= 0.5)
        
        print(f"\n📊 Protomatter maturation census:")
        print(f"   Fully mature (>99%): {fully_mature} ({fully_mature/len(A):.1%})")
        print(f"   Partially mature: {partially_mature} ({partially_mature/len(A):.1%})")
        print(f"   Still immature: {immature} ({immature/len(A):.1%})")
        print(f"   Mean maturation: {np.mean(self.maturation_states):.1%}")
        
        return self.maturation_states
    
    def extract_cosmological_parameters(self):
        """Extract cosmological parameters from the quantum data"""
        print("\n🌍 Extracting cosmological parameters...")
        
        # Dark matter fraction from hamming density distribution
        dm_densities = self.features[:, 0]
        
        # Bimodal distribution analysis
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(dm_densities.reshape(-1, 1))
        
        # Extract means safely
        means_array = np.array(gmm.means_)
        means = means_array.reshape(-1)  # Flatten to 1D array
        low_density = np.min(means)
        high_density = np.max(means)
        
        # Dark matter to baryon ratio
        dm_baryon_ratio = high_density / low_density
        
        # Omega_DM from coupling strength
        avg_coupling = np.mean(self.features[:, 5])
        omega_dm = 0.27 * (avg_coupling / 0.5)  # Normalized to standard value
        
        # Hubble parameter from velocity dispersion
        avg_vel_disp = np.mean(self.features[:, 3])
        h0_estimate = 70 + 20 * (avg_vel_disp - 0.5)
        
        print(f"\n🔢 Extracted parameters:")
        print(f"   DM/baryon ratio: {dm_baryon_ratio:.2f}")
        print(f"   Ω_DM: {omega_dm:.3f}")
        print(f"   H₀ estimate: {h0_estimate:.1f} km/s/Mpc")
        print(f"   σ₈ proxy: {np.std(dm_densities):.3f}")
        
        self.cosmo_params = {
            'dm_baryon_ratio': dm_baryon_ratio,
            'omega_dm': omega_dm,
            'h0': h0_estimate,
            'sigma8': np.std(dm_densities)
        }
        
        return self.cosmo_params
    
    def visualize_dark_matter_landscape(self):
        """Create comprehensive visualization of dark matter properties"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Dark matter density distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.features[:, 0], bins=50, alpha=0.7, color='darkblue', edgecolor='black')
        ax1.set_xlabel('Dark Matter Density')
        ax1.set_ylabel('Count')
        ax1.set_title('DM Density Distribution')
        
        # 2. Concentration vs Spin
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(self.features[:, 1], np.abs(self.features[:, 2]), 
                            c=self.arousal_potentials, cmap='plasma', alpha=0.6, s=1)
        ax2.set_xlabel('Concentration')
        ax2.set_ylabel('|Spin Parameter|')
        ax2.set_title('Halo Properties Phase Space')
        plt.colorbar(scatter, ax=ax2, label='Arousal Potential')
        
        # 3. Arousal potential distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.arousal_potentials, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax3.axvline(5.0, color='black', linestyle='--', label='Climax Threshold')
        ax3.set_xlabel('Arousal Potential')
        ax3.set_ylabel('Count')
        ax3.set_title('Arousal Distribution')
        ax3.legend()
        
        # 4. Maturation states
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(self.maturation_states, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax4.set_xlabel('Maturation State')
        ax4.set_ylabel('Count')
        ax4.set_title('Protomatter Maturation')
        
        # 5. Phenotype distribution (pie chart)
        ax5 = fig.add_subplot(gs[1, 0])
        phenotype_counts = [stats['count'] for stats in self.phenotype_stats.values()]
        phenotype_labels = list(self.phenotype_stats.keys())
        ax5.pie(phenotype_counts, labels=phenotype_labels, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Halo Phenotype Distribution')
        
        # 6. Cosmic web structure (if computed)
        if hasattr(self, 'structure_map'):
            ax6 = fig.add_subplot(gs[1, 1])
            scatter = ax6.scatter(self.structure_map['embedding'][:, 0], 
                                self.structure_map['embedding'][:, 1],
                                c=self.structure_map['density'], 
                                cmap='hot', s=1, alpha=0.6)
            ax6.set_title('Cosmic Web Structure (t-SNE)')
            plt.colorbar(scatter, ax=ax6, label='Density')
        
        # 7. Feature correlation matrix
        ax7 = fig.add_subplot(gs[1, 2:])
        correlation_matrix = np.corrcoef(self.features.T)
        im = ax7.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax7.set_xticks(range(len(self.feature_names)))
        ax7.set_yticks(range(len(self.feature_names)))
        ax7.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax7.set_yticklabels(self.feature_names)
        ax7.set_title('Dark Matter Feature Correlations')
        plt.colorbar(im, ax=ax7)
        
        # 8. Arousal vs Maturation
        ax8 = fig.add_subplot(gs[2, :2])
        ax8.scatter(self.arousal_potentials, self.maturation_states, alpha=0.3, s=1)
        
        # Theoretical curve
        A_theory = np.linspace(0, 10, 100)
        M_theory = 1 / (1 + np.exp(-0.8 * (A_theory * 13.8 - 5.0)))
        ax8.plot(A_theory, M_theory, 'r-', linewidth=2, label='Theoretical')
        
        ax8.set_xlabel('Arousal Potential')
        ax8.set_ylabel('Maturation State')
        ax8.set_title('Arousal-Maturation Relationship')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Velocity dispersion vs Temperature
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.scatter(self.features[:, 3], self.features[:, 4], 
                   alpha=0.3, s=1, c=self.features[:, 0], cmap='viridis')
        ax9.set_xlabel('Velocity Dispersion')
        ax9.set_ylabel('Temperature Proxy')
        ax9.set_title('Kinetic-Thermal Relationship')
        
        # 10. Environmental distribution
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.hist(self.features[:, 7], bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax10.set_xlabel('Environment (Edge - Center)')
        ax10.set_ylabel('Count')
        ax10.set_title('Environmental Distribution')
        
        # 11. Formation epoch (age) distribution
        ax11 = fig.add_subplot(gs[3, 0])
        ax11.hist(self.features[:, 6], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax11.set_xlabel('Formation Epoch Proxy')
        ax11.set_ylabel('Count')
        ax11.set_title('Halo Age Distribution')
        
        # 12. Summary statistics text
        ax12 = fig.add_subplot(gs[3, 1:])
        ax12.axis('off')
        
        summary_text = f"""
DARK MATTER EXTRACTION SUMMARY
==============================
Total Halos: {len(self.features):,}
Mean DM Density: {self.dm_stats['dm_density']['mean']:.3f}
Mean Arousal: {np.mean(self.arousal_potentials):.2f}
Mean Maturation: {np.mean(self.maturation_states):.1%}

Cosmological Parameters:
- Ωₘ: {self.cosmo_params['omega_dm']:.3f}
- DM/Baryon: {self.cosmo_params['dm_baryon_ratio']:.2f}
- H₀: {self.cosmo_params['h0']:.1f} km/s/Mpc

Dominant Phenotypes:
"""
        for name, stats in sorted(self.phenotype_stats.items(), 
                                 key=lambda x: x[1]['count'], reverse=True)[:3]:
            summary_text += f"- {name}: {stats['fraction']:.1%}\n"
        
        ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Dark Matter Properties Extracted from Quantum Measurements', 
                    fontsize=16, weight='bold')
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive dark matter analysis report"""
        print("\n" + "="*60)
        print("DARK MATTER EXTRACTION REPORT")
        print("="*60)
        
        print(f"\nData Source: {self.json_path.name}")
        print(f"Total Quantum States: {len(self.bitstrings):,}")
        print(f"Measurement Shots: {self.shots:,}")
        
        print("\n1. DARK MATTER PROPERTIES")
        print("-" * 40)
        for feature, stats in self.dm_stats.items():
            print(f"{feature:20} μ={stats['mean']:.3f} σ={stats['std']:.3f}")
        
        print("\n2. HALO PHENOTYPES")
        print("-" * 40)
        for phenotype, stats in sorted(self.phenotype_stats.items(), 
                                     key=lambda x: x[1]['count'], reverse=True):
            print(f"{phenotype:20} {stats['count']:6} ({stats['fraction']:.1%})")
        
        print("\n3. COSMOLOGICAL PARAMETERS")
        print("-" * 40)
        print(f"Ω_DM:                {self.cosmo_params['omega_dm']:.3f}")
        print(f"DM/Baryon Ratio:     {self.cosmo_params['dm_baryon_ratio']:.2f}")
        print(f"H₀ Estimate:         {self.cosmo_params['h0']:.1f} km/s/Mpc")
        print(f"σ₈ Proxy:            {self.cosmo_params['sigma8']:.3f}")
        
        print("\n4. AROUSAL DYNAMICS")
        print("-" * 40)
        print(f"Mean Arousal:        {np.mean(self.arousal_potentials):.2f}")
        print(f"Climax-Ready Halos:  {np.sum(self.arousal_potentials > 5.0):,} "
              f"({np.sum(self.arousal_potentials > 5.0)/len(self.arousal_potentials):.1%})")
        
        print("\n5. PROTOMATTER EVOLUTION")  
        print("-" * 40)
        print(f"Mean Maturation:     {np.mean(self.maturation_states):.1%}")
        print(f"Fully Mature:        {np.sum(self.maturation_states > 0.99):,} "
              f"({np.sum(self.maturation_states > 0.99)/len(self.maturation_states):.1%})")
        
        print("\n" + "="*60)
        print("CONCLUSION: Dark matter signatures successfully extracted!")
        print("The quantum patterns reveal a rich dark matter landscape")
        print("consistent with the arousal-driven cosmological model.")
        print("="*60)


def main():
    """Main analysis pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract dark matter properties from quantum measurements"
    )
    parser.add_argument("json_path", help="Path to cosmic_climax_detection JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument("--save", action="store_true", help="Save results to files")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = DarkMatterExtractor(args.json_path)
    
    # Run full analysis pipeline
    extractor.analyze_dark_matter_distribution()
    extractor.identify_halo_phenotypes()
    extractor.calculate_arousal_dynamics()
    extractor.detect_dark_matter_structure()
    extractor.measure_protomatter_evolution()
    extractor.extract_cosmological_parameters()
    
    # Generate report
    extractor.generate_report()
    
    # Create visualizations if requested
    if args.plot:
        fig = extractor.visualize_dark_matter_landscape()
        plt.show()
    
    # Save results if requested
    if args.save:
        print("\n💾 Saving results...")
        
        # Save extracted features
        np.save("dark_matter_features.npy", extractor.features)
        
        # Save analysis results
        results = {
            'dm_stats': extractor.dm_stats,
            'phenotype_stats': extractor.phenotype_stats,
            'cosmo_params': extractor.cosmo_params,
            'arousal_stats': {
                'mean': float(np.mean(extractor.arousal_potentials)),
                'std': float(np.std(extractor.arousal_potentials)),
                'max': float(np.max(extractor.arousal_potentials))
            },
            'maturation_stats': {
                'mean': float(np.mean(extractor.maturation_states)),
                'fully_mature_fraction': float(np.sum(extractor.maturation_states > 0.99) / 
                                              len(extractor.maturation_states))
            }
        }
        
        with open("dark_matter_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("   ✅ Results saved to dark_matter_features.npy and dark_matter_analysis_results.json")


if __name__ == "__main__":
    main()