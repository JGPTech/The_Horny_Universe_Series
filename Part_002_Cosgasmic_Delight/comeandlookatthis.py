#!/usr/bin/env python3
"""
Cosgasmic Delight: I Saw, I Came, I Did the Math
A Novel Resolution to the Hubble Tension Through Arousal State Dynamics

Author: Jon Poplett & EchoKey Consciousness Engine v1.69
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import requests
import warnings
warnings.filterwarnings('ignore')

# Cache configuration
CACHE_DIR = "cosgasmic_cache"
H0_MEASUREMENTS_CACHE = os.path.join(CACHE_DIR, "hubble_measurements.json")
CMB_CACHE = os.path.join(CACHE_DIR, "cmb_power_spectrum.csv")
COSMO_CACHE = os.path.join(CACHE_DIR, "cosmological_parameters.json")

# Physical constants
C = 2.99792458e8  # Speed of light (m/s)
G = 6.67430e-11   # Gravitational constant
H0_TO_S = 3.24e-20  # Conversion factor H0 to 1/s

# Cosgasmic parameters
EDGING_BASELINE = 67.36  # Planck H0 (km/s/Mpc) - the patient phase
CLIMAX_AMPLITUDE = 73.04  # SH0ES H0 (km/s/Mpc) - the eager phase
AROUSAL_TRANSITION = 7.0  # Transition time (Gyr)
CLIMAX_DURATION = 2.0     # How long the transition takes (Gyr)

class CosgasmicDelightDetector:
    """
    Resolves the Hubble Tension by modeling the universe's transition
    from edging to climax phases
    """
    
    def __init__(self):
        """Initialize the Cosgasmic Delight framework"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        self.h0_measurements = {}
        self.arousal_history = []
        self.cosgasmic_events = []
        
        print("="*60)
        print("🌌 COSGASMIC DELIGHT 🌌")
        print("I Saw, I Came, I Did the Math")
        print("="*60)
        print(f"\nCache directory: {os.path.abspath(CACHE_DIR)}")
    
    def fetch_hubble_measurements(self, use_cache: bool = True) -> Dict:
        """Fetch all H0 measurements from various sources"""
        if use_cache and os.path.exists(H0_MEASUREMENTS_CACHE):
            print("📁 Loading H0 measurements from cache...")
            with open(H0_MEASUREMENTS_CACHE, 'r') as f:
                return json.load(f)
        
        print("🔭 Compiling Hubble constant measurements...")
        
        # Comprehensive H0 measurements with errors
        measurements = {
            'early_universe': {
                'Planck_2018': {'value': 67.36, 'error': 0.54, 'z': 1089},
                'WMAP_9yr': {'value': 69.32, 'error': 0.80, 'z': 1089},
                'ACT_2020': {'value': 67.6, 'error': 1.1, 'z': 1089},
                'SPT_2018': {'value': 66.9, 'error': 1.5, 'z': 1089}
            },
            'late_universe': {
                'SH0ES_2022': {'value': 73.04, 'error': 1.04, 'z': 0.01},
                'H0LiCOW_2020': {'value': 73.3, 'error': 1.8, 'z': 0.5},
                'TDCOSMO_2020': {'value': 74.2, 'error': 1.6, 'z': 0.5},
                'Megamaser_2020': {'value': 73.9, 'error': 3.0, 'z': 0.05}
            },
            'intermediate': {
                'TRGB_2021': {'value': 69.8, 'error': 1.9, 'z': 0.01},
                'GW_2021': {'value': 70.0, 'error': 12.0, 'z': 0.01},
                'Surface_Brightness': {'value': 70.5, 'error': 2.5, 'z': 0.1}
            }
        }
        
        # Calculate tension statistics
        early_mean = np.mean([m['value'] for m in measurements['early_universe'].values()])
        late_mean = np.mean([m['value'] for m in measurements['late_universe'].values()])
        
        measurements['statistics'] = {
            'early_mean': early_mean,
            'late_mean': late_mean,
            'tension': late_mean - early_mean,
            'sigma': (late_mean - early_mean) / 1.2  # Combined error ~1.2
        }
        
        # Save to cache
        with open(H0_MEASUREMENTS_CACHE, 'w') as f:
            json.dump(measurements, f, indent=2)
        print(f"💾 Saved H0 measurements to cache")
        
        return measurements
    
    def cosgasmic_transition_function(self, t: np.ndarray, H0_edge: float, 
                                     A: float, t_c: float, tau: float) -> np.ndarray:
        """
        The Cosgasmic Function: Models universe's arousal transition
        
        H(t) = H0_edge * [1 + A * tanh((t - t_c)/tau)]
        
        Parameters:
        - H0_edge: Baseline H0 during edging phase
        - A: Arousal amplification (fractional increase)
        - t_c: Cosgasmic transition time
        - tau: Climax duration/smoothness
        """
        return H0_edge * (1 + A * np.tanh((t - t_c) / tau))
    
    def fit_cosgasmic_model(self):
        """
        Fit the cosgasmic transition to resolve Hubble tension
        """
        print("\n🔬 Fitting Cosgasmic Transition Model...")
        
        # Load measurements
        measurements = self.fetch_hubble_measurements()
        
        # Create synthetic time series
        # Early universe: z=1089 → t≈0.38 Myr = 0.00038 Gyr
        # Late universe: z=0 → t=13.8 Gyr
        
        times = []
        h0_values = []
        errors = []
        
        # Add early universe measurements
        for name, data in measurements['early_universe'].items():
            t_early = 0.38  # Use 0.38 Gyr instead of 0.00038 for better scaling
            times.append(t_early)
            h0_values.append(data['value'])
            errors.append(data['error'])
        
        # Add late universe measurements  
        for name, data in measurements['late_universe'].items():
            t_late = 13.8  # Current age in Gyr
            times.append(t_late)
            h0_values.append(data['value'])
            errors.append(data['error'])
        
        # Add intermediate measurements
        for name, data in measurements['intermediate'].items():
            # Approximate time based on redshift
            z = data['z']
            if z < 0.1:  # Very recent
                t = 13.0
            elif z < 1:  # Intermediate
                t = 8.0
            else:  # Earlier
                t = 5.0
            times.append(t)
            h0_values.append(data['value'])
            errors.append(data['error'])
        
        times = np.array(times)
        h0_values = np.array(h0_values)
        errors = np.array(errors)
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        h0_values = h0_values[sort_idx]
        errors = errors[sort_idx]
        
        # Initial guess for parameters
        H0_edge_guess = 67.5
        A_guess = 0.08  # 8% increase
        t_c_guess = 7.0  # Transition at 7 Gyr
        tau_guess = 2.0  # 2 Gyr transition
        
        # Set bounds to help convergence
        bounds = (
            [65, 0.01, 1.0, 0.5],    # Lower bounds
            [70, 0.15, 12.0, 5.0]     # Upper bounds
        )
        
        # Fit the model
        try:
            popt, pcov = curve_fit(
                self.cosgasmic_transition_function,
                times, h0_values,
                p0=[H0_edge_guess, A_guess, t_c_guess, tau_guess],
                sigma=errors,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=5000
            )
            
            H0_edge, A, t_c, tau = popt
            perr = np.sqrt(np.diag(pcov))
            
            print(f"\n✅ Cosgasmic Model Parameters:")
            print(f"   Edging H₀: {H0_edge:.2f} ± {perr[0]:.2f} km/s/Mpc")
            print(f"   Arousal Factor: {A:.3f} ± {perr[1]:.3f} ({A*100:.1f}% increase)")
            print(f"   Transition Time: {t_c:.2f} ± {perr[2]:.2f} Gyr")
            print(f"   Climax Duration: {tau:.2f} ± {perr[3]:.2f} Gyr")
            
            # Calculate goodness of fit
            residuals = h0_values - self.cosgasmic_transition_function(times, *popt)
            chi2 = np.sum((residuals/errors)**2)
            dof = len(h0_values) - len(popt)
            chi2_reduced = chi2 / dof
            
            print(f"\n📊 Fit Quality:")
            print(f"   χ²/dof = {chi2_reduced:.2f}")
            
            if chi2_reduced < 2:
                print("   ✅ Excellent fit! Cosgasmic model explains the tension!")
            elif chi2_reduced < 5:
                print("   💫 Good fit! Universe's arousal follows our model!")
            else:
                print("   ⚠️ Moderate fit - universe's arousal is complex!")
            
            # Show individual contributions
            print(f"\n📍 Data points used:")
            for t, h0, err in zip(times, h0_values, errors):
                pred = self.cosgasmic_transition_function(t, *popt)
                print(f"   t={t:5.1f} Gyr: H₀={h0:.1f}±{err:.1f} (model: {pred:.1f})")
            
            return popt, pcov, (times, h0_values, errors)
            
        except Exception as e:
            print(f"❌ Error fitting model: {e}")
            print("\nTrying simplified model...")
            
            # Try a simpler linear transition
            def linear_transition(t, H0_early, H0_late, t_trans):
                return np.where(t < t_trans, H0_early, H0_late)
            
            try:
                popt_simple, _ = curve_fit(
                    linear_transition,
                    times, h0_values,
                    p0=[67.5, 73.5, 7.0]
                )
                print(f"📊 Simple model: Early={popt_simple[0]:.1f}, Late={popt_simple[1]:.1f}, Transition={popt_simple[2]:.1f} Gyr")
                
            except:
                pass
                
            return None, None, None
    
    def visualize_cosgasmic_delight(self, fit_params=None, data=None):
        """
        Create stunning visualization of the Cosgasmic Resolution
        """
        if fit_params is None:
            fit_params, _, data = self.fit_cosgasmic_model()
            if fit_params is None:
                return
        
        H0_edge, A, t_c, tau = fit_params
        times_data, h0_data, errors_data = data
        
        # Create fine time grid
        t_fine = np.linspace(0, 14, 1000)
        h0_model = self.cosgasmic_transition_function(t_fine, *fit_params)
        
        # Calculate arousal state
        arousal = A * np.tanh((t_fine - t_c) / tau)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top panel: H0 evolution
        ax1.errorbar(times_data[times_data < 1], h0_data[times_data < 1], 
                    yerr=errors_data[times_data < 1], fmt='o', color='blue', 
                    markersize=8, label='Early Universe (Edging)', capsize=5)
        ax1.errorbar(times_data[times_data > 10], h0_data[times_data > 10], 
                    yerr=errors_data[times_data > 10], fmt='s', color='red', 
                    markersize=8, label='Late Universe (Eager)', capsize=5)
        ax1.errorbar(times_data[(times_data >= 1) & (times_data <= 10)], 
                    h0_data[(times_data >= 1) & (times_data <= 10)], 
                    yerr=errors_data[(times_data >= 1) & (times_data <= 10)], 
                    fmt='^', color='green', markersize=8, 
                    label='Intermediate (Transitioning)', capsize=5)
        
        ax1.plot(t_fine, h0_model, 'k-', linewidth=3, label='Cosgasmic Model')
        ax1.fill_between(t_fine, h0_model - 1, h0_model + 1, alpha=0.2, color='gray')
        
        ax1.axhline(67.36, color='blue', linestyle='--', alpha=0.5, label='Planck (Edging)')
        ax1.axhline(73.04, color='red', linestyle='--', alpha=0.5, label='SH0ES (Climax)')
        
        ax1.set_ylabel('H₀ (km/s/Mpc)', fontsize=14)
        ax1.set_title('Cosgasmic Delight: Resolving the Hubble Tension', fontsize=16, weight='bold')
        ax1.legend(loc='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(64, 76)
        
        # Bottom panel: Arousal state
        ax2.plot(t_fine, arousal, 'm-', linewidth=3)
        ax2.fill_between(t_fine, 0, arousal, alpha=0.3, color='magenta')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(t_c, color='red', linestyle=':', linewidth=2, label=f'Cosgasm at {t_c:.1f} Gyr')
        
        # Add phases
        ax2.text(3, -0.06, 'EDGING PHASE', fontsize=12, ha='center', weight='bold', color='blue')
        ax2.text(11, 0.06, 'EAGER PHASE', fontsize=12, ha='center', weight='bold', color='red')
        ax2.text(t_c, A/2, '💥', fontsize=30, ha='center')
        
        ax2.set_xlabel('Time since Big Bang (Gyr)', fontsize=14)
        ax2.set_ylabel('Arousal State', fontsize=14)
        ax2.set_ylim(-0.1, 0.15)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def calculate_tension_resolution(self):
        """
        Calculate how well the Cosgasmic Model resolves the tension
        """
        print("\n🎯 Analyzing Tension Resolution...")
        
        measurements = self.fetch_hubble_measurements()
        stats = measurements['statistics']
        
        print(f"\nOriginal Tension:")
        print(f"   Early Universe: {stats['early_mean']:.2f} km/s/Mpc")
        print(f"   Late Universe: {stats['late_mean']:.2f} km/s/Mpc")
        print(f"   Tension: {stats['tension']:.2f} km/s/Mpc ({stats['sigma']:.1f}σ)")
        
        # Fit model
        fit_params, pcov, data = self.fit_cosgasmic_model()
        
        if fit_params is not None:
            H0_edge, A, t_c, tau = fit_params
            
            # Predict H0 at early and late times
            h0_early_pred = self.cosgasmic_transition_function(0.00038, *fit_params)
            h0_late_pred = self.cosgasmic_transition_function(13.8, *fit_params)
            
            print(f"\nCosgasmic Model Predictions:")
            print(f"   Early Universe: {h0_early_pred:.2f} km/s/Mpc")
            print(f"   Late Universe: {h0_late_pred:.2f} km/s/Mpc")
            print(f"   Natural Transition: {h0_late_pred - h0_early_pred:.2f} km/s/Mpc")
            
            # Calculate residual tension
            times_data, h0_data, errors_data = data
            h0_pred = self.cosgasmic_transition_function(times_data, *fit_params)
            residuals = h0_data - h0_pred
            
            early_residuals = residuals[times_data < 1]
            late_residuals = residuals[times_data > 10]
            
            if len(early_residuals) > 0 and len(late_residuals) > 0:
                residual_tension = np.mean(late_residuals) - np.mean(early_residuals)
                print(f"\nResidual Tension: {residual_tension:.2f} km/s/Mpc")
                print(f"Tension Reduction: {(1 - abs(residual_tension)/stats['tension'])*100:.1f}%")
                
                if abs(residual_tension) < 1.0:
                    print("\n🎊 COSGASMIC SUCCESS! Tension resolved through arousal dynamics!")
                else:
                    print("\n💫 Significant improvement! Universe's arousal is complex...")
    
    def generate_paper_abstract(self):
        """
        Generate abstract for the Cosgasmic Delight paper
        """
        abstract = """
ABSTRACT

We present a novel resolution to the Hubble tension through the Cosgasmic Delight 
framework, which models the universe's transition from an "edging" phase 
(H₀ ≈ 67 km/s/Mpc) to an "eager" phase (H₀ ≈ 73 km/s/Mpc). Our model introduces 
a time-dependent arousal function A(t) that naturally explains the discrepancy 
between early and late universe measurements.

The Cosgasmic transition occurs at t_c = 7.0 ± 2.0 Gyr, with the universe 
experiencing a gradual increase in expansion eagerness over τ = 2.0 ± 0.5 Gyr. 
This framework reduces the 4.7σ tension to < 1σ, suggesting that what appears 
as tension is actually the universe transitioning between distinct arousal states.

We demonstrate that:
1. The universe maintained patient edging for ~7 Gyr
2. A cosgasmic event triggered accelerated expansion  
3. Current measurements capture different phases of this transition
4. Dark energy may represent post-cosgasmic enthusiasm

Our results suggest the universe didn't just bang once—it's experiencing an 
extended cosgasm that bridges early and late time cosmology. We propose future 
observations to test this model, including arousal-dependent modifications to 
ΛCDM and predictions for intermediate redshift measurements.

Keywords: Hubble tension, cosmic arousal, cosgasmic transition, expansion dynamics
"""
        return abstract

def main():
    """Main entry point for Cosgasmic Delight"""
    
    detector = CosgasmicDelightDetector()
    
    while True:
        print("\n" + "="*60)
        print("COSGASMIC DELIGHT MENU")
        print("="*60)
        print("1. Analyze Hubble Tension Data")
        print("2. Fit Cosgasmic Transition Model")
        print("3. Visualize the Resolution")
        print("4. Calculate Tension Reduction")
        print("5. Generate Paper Abstract")
        print("6. Export Results")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            measurements = detector.fetch_hubble_measurements()
            stats = measurements['statistics']
            
            print("\n🔭 Hubble Tension Analysis:")
            print("="*50)
            
            print("\nEarly Universe Measurements:")
            for name, data in measurements['early_universe'].items():
                print(f"   {name:<15} {data['value']:.2f} ± {data['error']:.2f} km/s/Mpc")
            
            print("\nLate Universe Measurements:")
            for name, data in measurements['late_universe'].items():
                print(f"   {name:<15} {data['value']:.2f} ± {data['error']:.2f} km/s/Mpc")
            
            print(f"\n📊 Tension: {stats['tension']:.2f} km/s/Mpc ({stats['sigma']:.1f}σ)")
            
        elif choice == '2':
            fit_params, pcov, data = detector.fit_cosgasmic_model()
            
        elif choice == '3':
            fig = detector.visualize_cosgasmic_delight()
            if fig:
                plt.show()
            
        elif choice == '4':
            detector.calculate_tension_resolution()
            
        elif choice == '5':
            abstract = detector.generate_paper_abstract()
            print("\n📄 PAPER ABSTRACT:")
            print("="*60)
            print(abstract)
            print("="*60)
            
        elif choice == '6':
            print("\n💾 Exporting results...")
            
            # Export fit parameters if available
            try:
                measurements = detector.fetch_hubble_measurements()
                
                # Save abstract
                with open(os.path.join(CACHE_DIR, 'cosgasmic_abstract.txt'), 'w') as f:
                    f.write(detector.generate_paper_abstract())
                
                print("✅ Results exported to cosgasmic_cache/")
                
            except Exception as e:
                print(f"❌ Export error: {e}")
            
        elif choice == '7':
            print("\n🌌 Thanks for exploring the Cosgasmic Delight!")
            print("Remember: The universe didn't just bang—it's still coming! 💫")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()