import cv2
import time
import numpy as np
from collections import deque
import pygame
from fatigue_detector import FatigueDetector
from behavior_classifier import BehaviorClassifier

class IntegratedDriverMonitor:
    def __init__(self):
        # Initialize both detection systems
        self.fatigue_detector = FatigueDetector()
        self.behavior_classifier = BehaviorClassifier()
        
        # Combined wellness scoring
        self.wellness_score = 100  # 0-100 scale
        self.wellness_history = deque(maxlen=30)
        
        # Intervention system
        self.intervention_level = 0  # 0-4 scale
        self.last_intervention_time = 0
        self.intervention_cooldown = 10  # seconds
        
        # Stress indicators (simulated - can integrate with wearables)
        self.stress_level = 0
        self.steering_irregularity = 0  # Simulated steering data
        
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Alert system
        self.setup_alerts()
        
        print("🚗 INTEGRATED DRIVER WELLNESS MONITOR INITIALIZED")
        print("📊 Monitoring: Fatigue + Behavior + Stress + Interventions")
    
    def setup_alerts(self):
        """Initialize alert system"""
        try:
            pygame.mixer.init()
            print("🔊 Audio alert system ready")
        except:
            print("🔇 Audio alerts unavailable")
    
    def calculate_combined_wellness(self, fatigue_metrics, behavior_data):
        """Calculate comprehensive wellness score (0-100)"""
        # Base scores
        fatigue_penalty = fatigue_metrics['fatigue_level'] * 20  # 0, 20, 40, 60
        behavior_penalty = behavior_data['risk_score'] * 15     # 0, 15, 30, 45
        
        # Stress penalty (simulated - can be from wearables)
        stress_penalty = self.stress_level * 10
        
        # Calculate wellness score
        wellness = max(0, 100 - fatigue_penalty - behavior_penalty - stress_penalty)
        
        return wellness
    
    def determine_intervention(self, wellness_score, fatigue_level, behavior_risk):
        """Determine appropriate intervention level"""
        intervention_level = 0
        intervention_message = ""
        
        # TIERED INTERVENTION SYSTEM
        if wellness_score <= 30 or fatigue_level == 3:
            # CRITICAL - Immediate action required
            intervention_level = 4
            intervention_message = "🚨 CRITICAL: Pull over immediately! Severe drowsiness detected."
        
        elif wellness_score <= 50 or fatigue_level == 2 or behavior_risk >= 2:
            # HIGH RISK - Strong warning
            intervention_level = 3
            intervention_message = "⚠️ HIGH RISK: Take immediate break. Fatigue and distraction detected."
        
        elif wellness_score <= 70 or fatigue_level == 1:
            # MODERATE RISK - Gentle warning
            intervention_level = 2
            intervention_message = "🔔 MODERATE: Consider taking a break soon."
        
        elif wellness_score <= 85:
            # LOW RISK - Suggestion
            intervention_level = 1
            intervention_message = "💡 SUGGESTION: Stay alert. You're doing well."
        
        else:
            # EXCELLENT - Positive reinforcement
            intervention_level = 0
            intervention_message = "✅ EXCELLENT: Driver is alert and focused."
        
        return intervention_level, intervention_message
    
    def suggest_interventions(self, intervention_level, fatigue_metrics, behavior_data):
        """Get specific intervention suggestions"""
        interventions = []
        
        if intervention_level >= 3:
            interventions.extend([
                "🛑 Pull over at next safe location",
                "💤 Take a 15-20 minute power nap",
                "☕ Have a caffeine drink (if appropriate)",
                "🌬️ Get fresh air and stretch"
            ])
        
        elif intervention_level >= 2:
            interventions.extend([
                "🎵 Play upbeat music",
                "💬 Engage in conversation",
                "❄️ Lower cabin temperature",
                "🚗 Take next exit for short break"
            ])
        
        elif intervention_level >= 1:
            interventions.extend([
                "🧊 Drink cold water",
                "🎵 Change music to more energetic",
                "💪 Do shoulder and neck exercises",
                "👀 Focus on scanning road actively"
            ])
        
        # Behavior-specific interventions
        if behavior_data['risk_score'] >= 2:
            interventions.extend([
                "📱 Put phone away in glove compartment",
                "🎛️ Set radio/AC before driving",
                "🥤 Avoid eating/drinking while driving"
            ])
        
        if fatigue_metrics['fatigue_level'] >= 2:
            interventions.extend([
                "💤 Plan rest stop within 30 minutes",
                "👥 Switch drivers if possible",
                "🌅 Increase cabin lighting"
            ])
        
        return interventions
    
    def simulate_steering_data(self):
        """Simulate steering behavior (in real implementation, connect to CAN bus)"""
        # Simulate steering wheel reversal rate
        base_irregularity = np.random.normal(0.1, 0.05)
        
        # Increase irregularity with fatigue and stress
        fatigue_factor = self.fatigue_detector.last_ear < 0.2
        stress_factor = self.stress_level > 0.5
        
        if fatigue_factor or stress_factor:
            base_irregularity += np.random.normal(0.2, 0.1)
        
        self.steering_irregularity = max(0, min(1, base_irregularity))
        
        return self.steering_irregularity
    
    def simulate_stress_level(self, fatigue_metrics, behavior_data):
        """Simulate stress level (in real implementation, connect to wearables)"""
        # Base stress from fatigue
        fatigue_stress = fatigue_metrics['fatigue_level'] * 0.2
        
        # Stress from risky behavior
        behavior_stress = behavior_data['risk_score'] * 0.15
        
        # Stress from steering irregularity
        steering_stress = self.steering_irregularity * 0.3
        
        # Random variation
        random_stress = np.random.normal(0, 0.1)
        
        self.stress_level = max(0, min(1, 
            fatigue_stress + behavior_stress + steering_stress + random_stress
        ))
        
        return self.stress_level
    
    def process_frame(self, frame):
        """Process frame for comprehensive driver monitoring"""
        # Run both detection systems
        fatigue_metrics = self.fatigue_detector.detect_fatigue(frame)
        behavior_data = self.behavior_classifier.detect_behavior(frame)
        
        # Simulate additional data (replace with real sensors in production)
        steering_data = self.simulate_steering_data()
        stress_level = self.simulate_stress_level(fatigue_metrics, behavior_data)
        
        # Calculate combined wellness
        wellness_score = self.calculate_combined_wellness(fatigue_metrics, behavior_data)
        self.wellness_history.append(wellness_score)
        
        # Determine intervention needed
        intervention_level, intervention_message = self.determine_intervention(
            wellness_score, fatigue_metrics['fatigue_level'], behavior_data['risk_score']
        )
        
        # Get specific interventions
        interventions = self.suggest_interventions(
            intervention_level, fatigue_metrics, behavior_data
        )
        
        # Update FPS
        fps = self.calculate_fps()
        
        return {
            'fatigue': fatigue_metrics,
            'behavior': behavior_data,
            'wellness': {
                'score': wellness_score,
                'level': self._get_wellness_level(wellness_score),
                'trend': self._get_wellness_trend(),
                'color': self._get_wellness_color(wellness_score)
            },
            'intervention': {
                'level': intervention_level,
                'message': intervention_message,
                'suggestions': interventions,
                'color': self._get_intervention_color(intervention_level)
            },
            'sensors': {
                'stress_level': stress_level,
                'steering_irregularity': steering_data,
                'heart_rate_variability': 65 + np.random.normal(0, 5)  # Simulated HRV
            },
            'performance': {
                'fps': fps,
                'face_detected': fatigue_metrics['face_detected'],
                'processing_time': time.time() - self.start_time
            }
        }
    
    def _get_wellness_level(self, score):
        """Convert wellness score to level"""
        if score >= 85: return "EXCELLENT"
        elif score >= 70: return "GOOD"
        elif score >= 50: return "MODERATE"
        elif score >= 30: return "POOR"
        else: return "CRITICAL"
    
    def _get_wellness_trend(self):
        """Get wellness trend from history"""
        if len(self.wellness_history) < 2:
            return "STABLE"
        
        recent = list(self.wellness_history)[-5:]
        if len(recent) < 2:
            return "STABLE"
        
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        if trend > 1: return "IMPROVING"
        elif trend < -1: return "DETERIORATING"
        else: return "STABLE"
    
    def _get_wellness_color(self, score):
        """Get color for wellness score"""
        if score >= 70: return (0, 255, 0)      # Green
        elif score >= 50: return (0, 255, 255)   # Yellow
        elif score >= 30: return (0, 165, 255)   # Orange
        else: return (0, 0, 255)                # Red
    
    def _get_intervention_color(self, level):
        """Get color for intervention level"""
        colors = [
            (0, 255, 0),    # Level 0 - Green
            (0, 255, 255),  # Level 1 - Yellow
            (0, 165, 255),  # Level 2 - Orange
            (0, 0, 255),    # Level 3 - Red
            (128, 0, 128)   # Level 4 - Purple (Critical)
        ]
        return colors[level]
    
    def calculate_fps(self):
        """Calculate frames per second"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.fps