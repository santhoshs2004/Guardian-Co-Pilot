import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import pygame

class FatigueDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # Lower for better detection
            min_tracking_confidence=0.5
        )
        
        # CORRECTED THRESHOLDS
        self.EAR_THRESHOLD = 0.20      # Eye closure threshold
        self.MAR_THRESHOLD = 0.50      # Yawn threshold
        self.EYE_CLOSED_ALERT = 0.8    # 800ms for alert
        self.EYE_CLOSED_SEVERE = 1.5   # 1.5 seconds for severe
        
        # State tracking
        self.eye_closed_start = None
        self.yawn_start = None
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
        
        # Debug info
        self.last_ear = 0.3
        self.last_mar = 0.3
        
        print("🚗 FATIGUE DETECTOR INITIALIZED")
        print(f"   EAR Threshold: {self.EAR_THRESHOLD}")
        print(f"   MAR Threshold: {self.MAR_THRESHOLD}")
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio with CORRECT landmarks"""
        try:
            # CORRECT MediaPipe eye landmarks for EAR calculation
            # Horizontal points
            p1 = eye_landmarks[0]  # Left corner
            p2 = eye_landmarks[1]  # Top center
            p3 = eye_landmarks[2]  # Right corner  
            p4 = eye_landmarks[3]  # Bottom center
            p5 = eye_landmarks[4]  # Left center
            p6 = eye_landmarks[5]  # Right center
            
            # Vertical distances
            A = np.linalg.norm(np.array(p2) - np.array(p6))
            B = np.linalg.norm(np.array(p3) - np.array(p5))
            # Horizontal distance
            C = np.linalg.norm(np.array(p1) - np.array(p4))
            
            ear = (A + B) / (2.0 * C)
            return ear
        except Exception as e:
            return 0.25
    
    def calculate_mar(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio with CORRECT landmarks"""
        try:
            # CORRECT MediaPipe mouth landmarks for MAR calculation
            # Outer mouth points
            p1 = mouth_landmarks[0]   # Left corner
            p2 = mouth_landmarks[1]   # Top center
            p3 = mouth_landmarks[2]   # Right corner
            p4 = mouth_landmarks[3]   # Bottom center
            
            # Vertical distance (mouth opening)
            vertical = np.linalg.norm(np.array(p2) - np.array(p4))
            # Horizontal distance (mouth width)
            horizontal = np.linalg.norm(np.array(p1) - np.array(p3))
            
            mar = vertical / horizontal
            return mar
        except Exception as e:
            return 0.3
    
    def detect_fatigue(self, frame):
        """Main detection function with CORRECT landmark mapping"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        current_time = time.time()
        
        # Default metrics
        metrics = {
            'ear': self.last_ear,
            'mar': self.last_mar,
            'fatigue_level': 0,
            'status': 'ALERT 🟢',
            'eye_state': 'OPEN',
            'mouth_state': 'CLOSED',
            'alert_message': '',
            'landmarks': [],
            'face_detected': False
        }
        
        if results.multi_face_landmarks:
            metrics['face_detected'] = True
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                landmarks = []
                
                # Extract all landmarks
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                
                # CORRECT MediaPipe landmark indices for eyes and mouth
                # Left eye landmarks (6 points in order)
                left_eye = [
                    landmarks[33],   # Left corner
                    landmarks[160],  # Top center  
                    landmarks[158],  # Right corner
                    landmarks[133],  # Bottom center
                    landmarks[153],  # Left center
                    landmarks[144]   # Right center
                ]
                
                # Right eye landmarks (6 points in order)  
                right_eye = [
                    landmarks[362],  # Right corner
                    landmarks[385],  # Top center
                    landmarks[387],  # Left corner  
                    landmarks[263],  # Bottom center
                    landmarks[373],  # Right center
                    landmarks[380]   # Left center
                ]
                
                # Mouth landmarks (4 key points)
                mouth = [
                    landmarks[61],   # Left corner
                    landmarks[13],   # Top center (upper lip)
                    landmarks[291],  # Right corner  
                    landmarks[14]    # Bottom center (lower lip)
                ]
                
                # Calculate EAR and MAR
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_mar(mouth)
                
                # Store for next frame
                self.last_ear = ear
                self.last_mar = mar
                
                # Immediate state detection
                eye_state = "CLOSED" if ear < self.EAR_THRESHOLD else "OPEN"
                mouth_state = "YAWNING" if mar > self.MAR_THRESHOLD else "CLOSED"
                
                # Debug output
                if int(time.time() * 2) % 5 == 0:  # Print every 2.5 seconds
                    print(f"EAR: {ear:.3f} ({eye_state}), MAR: {mar:.3f} ({mouth_state})")
                
                # Fatigue assessment
                fatigue_level, alert_message = self._assess_fatigue(
                    eye_state, mouth_state, current_time
                )
                
                # Trigger alerts
                if fatigue_level > 0:
                    self._play_alert(fatigue_level)
                
                metrics.update({
                    'ear': ear,
                    'mar': mar,
                    'fatigue_level': fatigue_level,
                    'status': self._get_fatigue_status(fatigue_level),
                    'eye_state': eye_state,
                    'mouth_state': mouth_state,
                    'alert_message': alert_message,
                    'landmarks': landmarks
                })
                
        else:
            metrics['face_detected'] = False
            metrics['alert_message'] = "No face detected"
        
        return metrics
    
    def _assess_fatigue(self, eye_state, mouth_state, current_time):
        """Fatigue assessment with timing"""
        fatigue_score = 0
        alert_message = ""
        
        # EYE CLOSURE DETECTION
        if eye_state == "CLOSED":
            if self.eye_closed_start is None:
                self.eye_closed_start = current_time
                alert_message = "Eyes closing detected..."
            else:
                eye_closed_duration = current_time - self.eye_closed_start
                
                if eye_closed_duration > self.EYE_CLOSED_SEVERE:
                    fatigue_score = 3
                    alert_message = f"SEVERE! Eyes closed {eye_closed_duration:.1f}s - PULL OVER!"
                elif eye_closed_duration > self.EYE_CLOSED_ALERT:
                    fatigue_score = 2
                    alert_message = f"Warning! Eyes closed {eye_closed_duration:.1f}s"
                else:
                    fatigue_score = 1
                    alert_message = "Slight drowsiness detected"
        else:
            if self.eye_closed_start is not None:
                # Just opened eyes
                closed_duration = current_time - self.eye_closed_start
                if closed_duration > 0.1:  # Only log meaningful closures
                    print(f"👁️ Eyes opened after {closed_duration:.1f}s")
                self.eye_closed_start = None
        
        # YAWN DETECTION
        if mouth_state == "YAWNING":
            if self.yawn_start is None:
                self.yawn_start = current_time
                print("😮 Yawn detected!")
            else:
                yawn_duration = current_time - self.yawn_start
                if yawn_duration > 2.0:
                    fatigue_score = max(fatigue_score, 2)
                    alert_message = "Frequent yawning - take a break!"
        else:
            if self.yawn_start is not None:
                self.yawn_start = None
        
        return fatigue_score, alert_message
    
    def _play_alert(self, level):
        """Play alert with cooldown"""
        current_time = time.time()
        if (current_time - self.last_alert_time) > self.alert_cooldown:
            if level == 1:
                print("🔔 Gentle alert: Feeling drowsy")
            elif level == 2:
                print("⚠️ WARNING: Moderate fatigue detected")
            elif level == 3:
                print("🚨 CRITICAL: SEVERE DROWSINESS - PULL OVER SAFELY!")
            
            self.last_alert_time = current_time
    
    def _get_fatigue_status(self, fatigue_level):
        status_map = {
            0: "ALERT 🟢",
            1: "SLIGHT DROWSINESS 🟡", 
            2: "MODERATE FATIGUE 🟠",
            3: "SEVERE DROWSINESS 🔴"
        }
        return status_map.get(fatigue_level, "ALERT 🟢")