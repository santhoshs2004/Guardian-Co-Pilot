import cv2
import numpy as np
import time
from collections import deque

class BehaviorClassifier:
    def __init__(self):
        self.behavior_classes = [
            'safe_driving', 'texting', 'phone_talking', 
            'drinking', 'reaching_behind', 'operating_radio'
        ]
        
        # Risk scores for each behavior
        self.risk_scores = {
            'safe_driving': 0,
            'texting': 3,
            'phone_talking': 2,
            'drinking': 2,
            'reaching_behind': 3,
            'operating_radio': 1
        }
        
        # Behavior tracking with time-based decay
        self.behavior_history = deque(maxlen=10)
        self.current_behavior = 'safe_driving'
        self.behavior_start_time = time.time()
        self.min_behavior_duration = 2.0  # Minimum time to change behavior
        
        # Motion detection
        self.prev_frame = None
        self.motion_threshold = 1500
        self.motion_history = deque(maxlen=15)
        
        # Region activity tracking
        self.face_activity = deque(maxlen=10)
        self.left_activity = deque(maxlen=10)
        self.right_activity = deque(maxlen=10)
        
        print("🚦 IMPROVED BEHAVIOR CLASSIFIER INITIALIZED")
        print(f"   Monitoring: {', '.join(self.behavior_classes)}")
        print("   Using: Smart motion analysis + Time-based detection")
    
    def detect_behavior(self, frame):
        """Detect driver behavior with improved real-time responsiveness"""
        h, w = frame.shape[:2]
        
        # Extract features with smoothing
        features = self._extract_smart_features(frame)
        
        # Classify behavior
        detected_behavior, confidence = self._classify_behavior_smart(features, frame)
        
        # Time-based behavior switching (prevent rapid changes)
        current_time = time.time()
        time_since_change = current_time - self.behavior_start_time
        
        # Only change behavior if it's been stable for minimum duration
        if (detected_behavior != self.current_behavior and 
            time_since_change > self.min_behavior_duration and 
            confidence > 0.6):
            
            self.current_behavior = detected_behavior
            self.behavior_start_time = current_time
            print(f"🔄 Behavior changed to: {detected_behavior} (Confidence: {confidence:.1f})")
        
        # Update history
        self.behavior_history.append(self.current_behavior)
        
        # Print status occasionally
        if int(time.time()) % 4 == 0:  # Every 4 seconds
            motion_avg = np.mean(list(self.motion_history)) if self.motion_history else 0
            print(f"📊 Current: {self.current_behavior} | Motion: {motion_avg:.0f} | Face: {np.mean(list(self.face_activity)):.1f}")
        
        return {
            'behavior': self.current_behavior,
            'confidence': confidence,
            'risk_score': self.risk_scores[self.current_behavior],
            'features': features,
            'hands_detected': features.get('left_region_active', False) or features.get('right_region_active', False),
            'face_detected': True,
            'hand_count': features.get('hand_like_objects', 0)
        }
    
    def _extract_smart_features(self, frame):
        """Extract features with motion smoothing and region analysis"""
        h, w = frame.shape[:2]
        features = {
            'motion_intensity': 0,
            'hand_like_objects': 0,
            'face_region_active': 0,
            'left_region_active': 0,
            'right_region_active': 0,
            'recent_motion_avg': 0
        }
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Initialize previous frame if needed
        if self.prev_frame is None:
            self.prev_frame = gray
            return features
        
        # Motion detection with better thresholding
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Count motion pixels
        motion_pixels = cv2.countNonZero(thresh)
        features['motion_intensity'] = motion_pixels
        
        # Update motion history
        self.motion_history.append(motion_pixels)
        features['recent_motion_avg'] = np.mean(list(self.motion_history)) if self.motion_history else 0
        
        # Find meaningful contours (ignore small noise)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_like_objects = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 800 < area < 20000:  # Meaningful object sizes
                hand_like_objects += 1
                # Draw contours for visualization
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        features['hand_like_objects'] = hand_like_objects
        
        self.prev_frame = gray
        
        # Analyze regions with activity scoring (0-1)
        face_activity = self._analyze_region_activity(frame, w//3, 0, 2*w//3, h//3)
        left_activity = self._analyze_region_activity(frame, 0, h//2, w//4, h)
        right_activity = self._analyze_region_activity(frame, 3*w//4, h//2, w, h)
        
        # Update activity histories
        self.face_activity.append(face_activity)
        self.left_activity.append(left_activity)
        self.right_activity.append(right_activity)
        
        # Use smoothed activity values
        features['face_region_active'] = np.mean(list(self.face_activity))
        features['left_region_active'] = np.mean(list(self.left_activity))
        features['right_region_active'] = np.mean(list(self.right_activity))
        
        # Draw regions with activity levels
        self._draw_smart_regions(frame, features, w, h)
        
        return features
    
    def _analyze_region_activity(self, frame, x1, y1, x2, y2):
        """Analyze activity level in a region (0-1 scale)"""
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        
        # Calculate color variation (indicates movement/changes)
        color_std = np.std(region)
        
        # Normalize to 0-1 scale
        activity_level = min(1.0, color_std / 50.0)
        
        return activity_level
    
    def _draw_smart_regions(self, frame, features, w, h):
        """Draw regions with activity-based coloring"""
        # Face region (green, intensity based on activity)
        face_color = (0, int(255 * features['face_region_active']), 0)
        cv2.rectangle(frame, (w//3, 0), (2*w//3, h//3), face_color, 3)
        cv2.putText(frame, f"FACE: {features['face_region_active']:.1f}", 
                   (w//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
        
        # Left hand region (blue, intensity based on activity)
        left_color = (int(255 * features['left_region_active']), 0, 0)
        cv2.rectangle(frame, (0, h//2), (w//4, h), left_color, 3)
        cv2.putText(frame, f"LEFT: {features['left_region_active']:.1f}", 
                   (10, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        
        # Right hand region (red, intensity based on activity)
        right_color = (0, 0, int(255 * features['right_region_active']))
        cv2.rectangle(frame, (3*w//4, h//2), (w, h), right_color, 3)
        cv2.putText(frame, f"RIGHT: {features['right_region_active']:.1f}", 
                   (3*w//4 + 10, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        
        # Motion indicator
        motion_level = min(1.0, features['motion_intensity'] / 5000.0)
        motion_color = (0, int(255 * motion_level), int(255 * (1 - motion_level)))
        cv2.putText(frame, f"MOTION: {features['motion_intensity']}", 
                   (w//2 - 50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 2)
    
    def _classify_behavior_smart(self, features, frame):
        """Improved behavior classification with better logic"""
        behavior = 'safe_driving'
        confidence = 1.0
        
        # Activity thresholds (0-1 scale)
        face_active = features['face_region_active'] > 0.3
        left_active = features['left_region_active'] > 0.3
        right_active = features['right_region_active'] > 0.3
        high_motion = features['recent_motion_avg'] > 2000
        
        # BEHAVIOR DECISION TREE
        
        # 1. Texting (both hands active, looking down/away)
        if left_active and right_active and not face_active and high_motion:
            behavior = 'texting'
            confidence = 0.9
        
        # 2. Phone talking (one hand near face, face active)
        elif (left_active or right_active) and face_active and not high_motion:
            behavior = 'phone_talking'
            confidence = 0.8
        
        # 3. Operating radio (face active, hands inactive)
        elif face_active and not left_active and not right_active:
            behavior = 'operating_radio'
            confidence = 0.7
        
        # 4. Drinking (face active + one hand active, moderate motion)
        elif face_active and (left_active or right_active) and features['motion_intensity'] > 1000:
            behavior = 'drinking'
            confidence = 0.6
        
        # 5. Reaching behind (high motion, multiple objects)
        elif high_motion and features['hand_like_objects'] >= 2:
            behavior = 'reaching_behind'
            confidence = 0.7
        
        # 6. Safe driving (low activity in all regions)
        elif not face_active and not left_active and not right_active and not high_motion:
            behavior = 'safe_driving'
            confidence = 0.9
        
        return behavior, confidence
    
    def get_behavior_description(self, behavior):
        """Get description for each behavior"""
        descriptions = {
            'safe_driving': "Normal driving detected ✅",
            'texting': "⚠️ TEXTING DETECTED - Both hands active!",
            'phone_talking': "📱 PHONE DETECTED - Hand near face",
            'drinking': "🥤 DRINKING DETECTED - Hand to mouth", 
            'reaching_behind': "🔄 REACHING DETECTED - High motion",
            'operating_radio': "🎛️ RADIO DETECTED - Head movement"
        }
        return descriptions.get(behavior, "Monitoring driver behavior")