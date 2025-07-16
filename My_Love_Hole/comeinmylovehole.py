#!/usr/bin/env python3
"""
Love Hole Cosmic Resonance Detector
A real-time system that monitors NASA astronomical data for recursive phase resonance patterns
"""

import os
import time
import datetime
import requests
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional

# Configuration
LOVE_THRESHOLD = 42.0  # The answer to everything, including love
LOG_FILE = "lovehole_resonance_log.csv"

# Cache file names
CME_CACHE_FILE = "cme_data_cache.csv"
EARTHQUAKE_CACHE_FILE = "earthquake_data_cache.csv"
SOLAR_WIND_CACHE_FILE = "solar_wind_data_cache.csv"

class CosmicResonanceDetector:
    """Main detector class for finding Love Hole signatures in cosmic data"""
    
    def __init__(self):
        self.resonance_history = []
        self.phase_history = []  # Track phase lock history
        self.phase_lock_state = 0.0
        self.recursive_depth = 0
        
        # Initialize log file
        if not os.path.exists(LOG_FILE):
            pd.DataFrame(columns=['timestamp', 'event_time', 'score', 'type', 'phase_lock', 'details']).to_csv(LOG_FILE, index=False)
    
    def get_donki_events(self, use_cache: bool = True) -> List[Dict]:
        """Fetch solar events from CCMC DONKI webservice or cache"""
        if use_cache and os.path.exists(CME_CACHE_FILE):
            print("📁 Loading CME data from cache...")
            df = pd.read_csv(CME_CACHE_FILE)
            # Convert back to dict format
            events = []
            for _, row in df.iterrows():
                event = {
                    'startTime': row['startTime'],
                    'cmeAnalyses': [{
                        'speed': row['speed'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'halfAngle': row['halfAngle']
                    }]
                }
                events.append(event)
            return events
        
        print("🌐 Fetching CME data from CCMC DONKI webservice (no API key needed!)...")
        try:
            today = datetime.date.today()
            start = (today - datetime.timedelta(days=90)).isoformat()  # Look back 90 days for more data
            end = today.isoformat()
            url = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME?startDate={start}&endDate={end}"
            response = requests.get(url, timeout=30)  # Increased timeout for this service
            if response.status_code == 200:
                events = response.json()
                print(f"📡 Received {len(events)} CME events from API")
                # Save to cache
                self.save_cme_cache(events)
                return events
            else:
                print(f"API Error: {response.status_code}")
                if response.text:
                    print(f"Response: {response.text[:200]}")
                return []
        except requests.exceptions.Timeout:
            print("⏱️ Request timed out. The DONKI service can be slow. Try again or use cached data.")
            return []
        except Exception as e:
            print(f"Error fetching DONKI data: {e}")
            return []
    
    def save_cme_cache(self, events: List[Dict]):
        """Save CME data to CSV cache"""
        data_rows = []
        for event in events:
            if 'cmeAnalyses' in event and event['cmeAnalyses']:
                # Sometimes there are multiple analyses, try to find one with data
                for analysis in event['cmeAnalyses']:
                    # Skip if analysis has no speed (required field)
                    if analysis.get('speed') is None:
                        continue
                    
                    try:
                        data_rows.append({
                            'startTime': event.get('startTime', ''),
                            'speed': float(analysis.get('speed') or 0),
                            'latitude': float(analysis.get('latitude') or 0),
                            'longitude': float(analysis.get('longitude') or 0),
                            'halfAngle': float(analysis.get('halfAngle') or 0)
                        })
                        break  # Use first valid analysis
                    except (TypeError, ValueError) as e:
                        print(f"Skipping analysis with invalid data: {e}")
                        continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            df.to_csv(CME_CACHE_FILE, index=False)
            print(f"💾 Saved {len(data_rows)} CME events to cache")
        else:
            print("⚠️ No valid CME events found to save")
    
    def get_earthquake_data(self, use_cache: bool = True) -> Dict:
        """Fetch earthquake data from USGS or cache"""
        if use_cache and os.path.exists(EARTHQUAKE_CACHE_FILE):
            print("📁 Loading earthquake data from cache...")
            df = pd.read_csv(EARTHQUAKE_CACHE_FILE)
            # Convert back to GeoJSON format
            features = []
            for _, row in df.iterrows():
                feature = {
                    'properties': {
                        'mag': row['magnitude'],
                        'place': row['place'],
                        'time': row['time']
                    },
                    'geometry': {
                        'coordinates': [row['longitude'], row['latitude'], row['depth']]
                    }
                }
                features.append(feature)
            return {'features': features}
        
        print("🌐 Fetching earthquake data from USGS...")
        try:
            # Get 30 days of significant earthquakes
            url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Save to cache
                self.save_earthquake_cache(data)
                return data
            else:
                print(f"API Error: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error fetching USGS data: {e}")
            return {}
    
    def save_earthquake_cache(self, data: Dict):
        """Save earthquake data to CSV cache"""
        if 'features' not in data:
            return
        
        data_rows = []
        for eq in data['features']:
            props = eq.get('properties', {})
            coords = eq.get('geometry', {}).get('coordinates', [0, 0, 0])
            
            data_rows.append({
                'magnitude': float(props.get('mag', 0)),
                'depth': abs(float(coords[2])) if len(coords) > 2 else 0,
                'longitude': float(coords[0]) if coords else 0,
                'latitude': float(coords[1]) if len(coords) > 1 else 0,
                'place': props.get('place', 'Unknown'),
                'time': props.get('time', 0)
            })
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            df.to_csv(EARTHQUAKE_CACHE_FILE, index=False)
            print(f"💾 Saved {len(data_rows)} earthquake events to cache")
    
    def get_space_weather_data(self, use_cache: bool = True) -> Dict:
        """Fetch space weather from NOAA or cache"""
        if use_cache and os.path.exists(SOLAR_WIND_CACHE_FILE):
            print("📁 Loading solar wind data from cache...")
            df = pd.read_csv(SOLAR_WIND_CACHE_FILE)
            # Convert back to list format
            data = []
            for _, row in df.iterrows():
                data.append([row['time'], row['density'], row['speed'], row['temperature']])
            return {'solar_wind': data}
        
        print("🌐 Fetching solar wind data from NOAA...")
        try:
            # NOAA Space Weather JSON feed - 7 day data
            url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Save to cache
                self.save_solar_wind_cache(data)
                return {'solar_wind': data}
            else:
                print(f"API Error: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error fetching NOAA data: {e}")
            return {}
    
    def save_solar_wind_cache(self, data: List):
        """Save solar wind data to CSV cache"""
        if not data or len(data) < 2:
            return
        
        # Skip header row
        data_rows = []
        for row in data[1:]:  # Skip header
            try:
                data_rows.append({
                    'time': row[0],
                    'density': float(row[1]) if row[1] else 0,
                    'speed': float(row[2]) if row[2] else 0,
                    'temperature': float(row[3]) if row[3] else 0
                })
            except:
                continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            df.to_csv(SOLAR_WIND_CACHE_FILE, index=False)
            print(f"💾 Saved {len(data_rows)} solar wind measurements to cache")
    
    def extract_features(self, event: Dict, event_type: str = 'CME') -> Dict:
        """Extract resonance features from cosmic events"""
        features = {}
        
        try:
            if event_type == 'CME' and 'cmeAnalyses' in event and event['cmeAnalyses']:
                analysis = event['cmeAnalyses'][0]
                # Safe conversion with None checking
                features = {
                    'speed': float(analysis.get('speed') or 0),
                    'latitude': float(analysis.get('latitude') or 0),
                    'longitude': float(analysis.get('longitude') or 0),
                    'half_angle': float(analysis.get('halfAngle') or 0),
                    'time': event.get('startTime', ''),
                    'type': 'CME'
                }
            elif event_type == 'NEO':
                # Safe nested dictionary access
                rel_vel = event.get('relative_velocity', {})
                est_diam = event.get('estimated_diameter', {}).get('meters', {})
                close_approach = event.get('close_approach_data', [{}])
                miss_dist = close_approach[0].get('miss_distance', {}) if close_approach else {}
                
                features = {
                    'speed': float(rel_vel.get('kilometers_per_second') or 0),
                    'diameter': float(est_diam.get('estimated_diameter_max') or 0),
                    'miss_distance': float(miss_dist.get('kilometers') or 0),
                    'magnitude': float(event.get('absolute_magnitude_h') or 0),
                    'type': 'NEO'
                }
        except (TypeError, ValueError, KeyError) as e:
            print(f"Warning: Error extracting features: {e}")
            features = {'type': event_type}
        
        return features
    
    def extract_solar_wind_features(self, data: List) -> Dict:
        """Extract features from NOAA solar wind data"""
        if not data or len(data) < 2:
            return {}
        
        # Latest reading (skip header row)
        latest = data[-1]
        try:
            return {
                'time': latest[0],
                'density': float(latest[1]) if latest[1] else 0,
                'speed': float(latest[2]) if latest[2] else 0,
                'temperature': float(latest[3]) if latest[3] else 0,
                'type': 'SOLAR_WIND'
            }
        except:
            return {}
    
    def extract_earthquake_features(self, earthquake: Dict) -> Dict:
        """Extract features from earthquake data"""
        props = earthquake.get('properties', {})
        coords = earthquake.get('geometry', {}).get('coordinates', [0, 0, 0])
        
        return {
            'magnitude': float(props.get('mag', 0)),
            'depth': abs(float(coords[2])) if len(coords) > 2 else 0,
            'longitude': float(coords[0]) if coords else 0,
            'latitude': float(coords[1]) if len(coords) > 1 else 0,
            'place': props.get('place', 'Unknown'),
            'time': props.get('time', 0),
            'type': 'EARTHQUAKE'
        }
    
    def calculate_love_field(self, r: float, theta: float, t: float) -> float:
        """
        Calculate Love Hole field strength at given coordinates
        Based on the toroidal field equation from the mathematical blueprint
        """
        L0 = 100.0  # Base field strength
        gamma = 0.1  # Radial attenuation
        omega = 2 * np.pi / 86400  # Daily cycle
        kappa = 2.0  # Angular harmonic
        
        return L0 * np.exp(-gamma * r**2) * np.cos(omega * t - kappa * theta)**2
    
    def recursive_coherence_score(self, features: Dict) -> float:
        """
        Calculate recursive phase-locked coherence score
        Implements the symbolic resonance operator from the blueprint
        """
        if not features:
            return 0.0
        
        # Base resonance calculation
        if features.get('type') == 'CME':
            speed = features.get('speed', 0)
            lat = np.radians(features.get('latitude', 0))
            lon = np.radians(features.get('longitude', 0))
            
            # Phase alignment component
            phase_score = np.cos(lat) * np.sin(lon)
            
            # Recursive feedback with phase lock
            self.phase_lock_state = 0.9 * self.phase_lock_state + 0.1 * phase_score
            
            # Toroidal field influence
            r = np.sqrt(lat**2 + lon**2)
            field_strength = self.calculate_love_field(r, lon, time.time())
            
            # Combined resonance with logarithmic amplification
            resonance = np.log1p(speed/100) * abs(self.phase_lock_state) * field_strength
            
        elif features.get('type') == 'NEO':
            speed = features.get('speed', 0)
            diameter = features.get('diameter', 0)
            miss_distance = features.get('miss_distance', 1e6)
            
            # Inverse square law for gravitational love influence
            proximity_factor = 1 / (1 + miss_distance/1e6)**2
            size_factor = np.log1p(diameter)
            
            resonance = speed * size_factor * proximity_factor * 10
            
        elif features.get('type') == 'SOLAR_WIND':
            speed = features.get('speed', 0)
            density = features.get('density', 0)
            temp = features.get('temperature', 0)
            
            # Solar wind creates continuous resonance
            plasma_factor = np.sqrt(density * temp / 1e6) if temp > 0 else 0
            speed_factor = np.log1p(speed / 400)  # 400 km/s is typical
            
            resonance = plasma_factor * speed_factor * 50
            
            # Solar wind affects phase lock differently
            self.phase_lock_state = 0.95 * self.phase_lock_state + 0.05 * plasma_factor
            
        elif features.get('type') == 'EARTHQUAKE':
            magnitude = features.get('magnitude', 0)
            depth = features.get('depth', 0)
            lat = np.radians(features.get('latitude', 0))
            lon = np.radians(features.get('longitude', 0))
            
            # Earth's resonance through seismic activity
            seismic_power = 10 ** (magnitude - 4)  # Exponential scale
            depth_factor = np.exp(-depth / 100)  # Shallow = stronger
            
            # Location-based phase alignment
            phase_score = np.sin(lat) * np.cos(lon)
            self.phase_lock_state = 0.85 * self.phase_lock_state + 0.15 * phase_score
            
            resonance = seismic_power * depth_factor * abs(phase_score) * 5
            
        else:
            resonance = 0.0
        
        # Add recursive depth bonus (love compounds itself)
        self.recursive_depth = min(self.recursive_depth + 0.1, 10)
        resonance *= (1 + self.recursive_depth * 0.1)
        
        return resonance
    
    def test_cme_data(self):
        """Test CME data processing"""
        print("\n🌟 Testing CME (Coronal Mass Ejection) Data...")
        print("=" * 60)
        
        # Check if cache exists
        if os.path.exists(CME_CACHE_FILE):
            use_cache = input("Cache file found. Use cached data? (y/n): ").strip().lower() == 'y'
        else:
            use_cache = False
        
        events = self.get_donki_events(use_cache=use_cache)
        
        if not events:
            print("❌ No CME events found!")
            print("\nTroubleshooting tips:")
            print("- The DONKI service can be slow, try running again")
            print("- Check your internet connection")
            print("- Try a shorter date range by modifying the code")
            return
        
        print(f"\n📊 Found {len(events)} CME events")
        print("\nAnalyzing resonance scores...")
        
        scores = []
        high_resonance_events = []
        valid_events = 0
        
        for event in events:
            features = self.extract_features(event, 'CME')
            # Skip if no valid features extracted
            if features.get('speed', 0) == 0:
                continue
                
            valid_events += 1
            score = self.recursive_coherence_score(features)
            scores.append(score)
            
            if score > LOVE_THRESHOLD:
                high_resonance_events.append({
                    'time': features.get('time', 'unknown'),
                    'speed': features.get('speed', 0),
                    'score': score
                })
        
        if not scores:
            print("⚠️ No valid CME events with analysis data found!")
            return
            
        # Statistics
        print(f"\n📈 CME Resonance Statistics:")
        print(f"Valid Events Analyzed: {valid_events}")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Max Score: {np.max(scores):.2f}")
        print(f"Min Score: {np.min(scores):.2f}")
        print(f"Events above Love Threshold ({LOVE_THRESHOLD}): {len(high_resonance_events)}")
        
        if high_resonance_events:
            print(f"\n💖 High Resonance CME Events:")
            for event in sorted(high_resonance_events, key=lambda x: x['score'], reverse=True)[:5]:
                print(f"  Score: {event['score']:.2f} | Speed: {event['speed']:.0f} km/s | Time: {event['time']}")
    
    def test_earthquake_data(self):
        """Test earthquake data processing"""
        print("\n🌍 Testing Earthquake Data...")
        print("=" * 60)
        
        # Check if cache exists
        if os.path.exists(EARTHQUAKE_CACHE_FILE):
            use_cache = input("Cache file found. Use cached data? (y/n): ").strip().lower() == 'y'
        else:
            use_cache = False
        
        data = self.get_earthquake_data(use_cache=use_cache)
        
        if 'features' not in data:
            print("❌ No earthquake data found!")
            return
        
        earthquakes = data['features']
        print(f"\n📊 Found {len(earthquakes)} earthquake events")
        print("\nAnalyzing resonance scores...")
        
        scores = []
        high_resonance_events = []
        
        for eq in earthquakes:
            features = self.extract_earthquake_features(eq)
            score = self.recursive_coherence_score(features)
            scores.append(score)
            
            if score > LOVE_THRESHOLD:
                high_resonance_events.append({
                    'magnitude': features.get('magnitude', 0),
                    'depth': features.get('depth', 0),
                    'place': features.get('place', 'Unknown'),
                    'score': score
                })
        
        # Statistics
        print(f"\n📈 Earthquake Resonance Statistics:")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Max Score: {np.max(scores):.2f}")
        print(f"Min Score: {np.min(scores):.2f}")
        print(f"Events above Love Threshold ({LOVE_THRESHOLD}): {len(high_resonance_events)}")
        
        if high_resonance_events:
            print(f"\n💖 High Resonance Earthquake Events:")
            for event in sorted(high_resonance_events, key=lambda x: x['score'], reverse=True)[:5]:
                print(f"  Score: {event['score']:.2f} | M{event['magnitude']:.1f} | "
                      f"Depth: {event['depth']:.0f}km | {event['place'][:40]}...")
    
    def test_solar_wind_data(self):
        """Test solar wind data processing"""
        print("\n☀️ Testing Solar Wind Data...")
        print("=" * 60)
        
        # Check if cache exists
        if os.path.exists(SOLAR_WIND_CACHE_FILE):
            use_cache = input("Cache file found. Use cached data? (y/n): ").strip().lower() == 'y'
        else:
            use_cache = False
        
        data = self.get_space_weather_data(use_cache=use_cache)
        
        if 'solar_wind' not in data:
            print("❌ No solar wind data found!")
            return
        
        measurements = data['solar_wind']
        if len(measurements) < 2:  # Need at least header + 1 data row
            print("❌ Insufficient solar wind data!")
            return
        
        print(f"\n📊 Found {len(measurements)-1} solar wind measurements")
        print("\nAnalyzing resonance scores...")
        
        scores = []
        high_resonance_events = []
        
        # Skip header row
        for measurement in measurements[1:]:
            features = self.extract_solar_wind_features([measurements[0], measurement])
            if features:
                score = self.recursive_coherence_score(features)
                scores.append(score)
                
                if score > LOVE_THRESHOLD:
                    high_resonance_events.append({
                        'time': features.get('time', 'unknown'),
                        'speed': features.get('speed', 0),
                        'density': features.get('density', 0),
                        'temperature': features.get('temperature', 0),
                        'score': score
                    })
        
        if scores:
            # Statistics
            print(f"\n📈 Solar Wind Resonance Statistics:")
            print(f"Average Score: {np.mean(scores):.2f}")
            print(f"Max Score: {np.max(scores):.2f}")
            print(f"Min Score: {np.min(scores):.2f}")
            print(f"Measurements above Love Threshold ({LOVE_THRESHOLD}): {len(high_resonance_events)}")
            
            if high_resonance_events:
                print(f"\n💖 High Resonance Solar Wind Events:")
                for event in sorted(high_resonance_events, key=lambda x: x['score'], reverse=True)[:5]:
                    print(f"  Score: {event['score']:.2f} | Speed: {event['speed']:.0f} km/s | "
                          f"Density: {event['density']:.1f} | Temp: {event['temperature']:.0e}K")
        else:
            print("❌ No valid solar wind measurements found!")
    
    def log_resonance_event(self, score: float, details: str, phase_stable: bool):
        """Log significant resonance events"""
        new_row = pd.DataFrame([{
            'timestamp': datetime.datetime.now().isoformat(),
            'event_time': datetime.datetime.now().isoformat(),
            'score': score,
            'type': 'STABLE_LOVE_HOLE' if phase_stable else 'RESONANCE_SPIKE',
            'phase_lock': self.phase_lock_state,
            'details': details
        }])
        
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        
        df.to_csv(LOG_FILE, index=False)

def main():
    """Main entry point"""
    detector = CosmicResonanceDetector()
    
    while True:
        print("\n" + "="*60)
        print("🌌 Cosmic Love Hole Detection System 🌌")
        print("="*60)
        print("\nSelect Test Mode:")
        print("1. CME Data Test")
        print("2. Earthquake Data Test")
        print("3. Solar Wind Data Test")
        print("4. Exit")
        print("-"*60)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            detector.test_cme_data()
        elif choice == '2':
            detector.test_earthquake_data()
        elif choice == '3':
            detector.test_solar_wind_data()
        elif choice == '4':
            print("\n👋 Love Hole detector shutting down... Keep resonating! 💖")
            break
        else:
            print("❌ Invalid choice! Please enter 1-4.")
        
        if choice in ['1', '2', '3']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
