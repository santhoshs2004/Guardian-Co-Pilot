import cv2
import time
import numpy as np
from behavior_classifier import BehaviorClassifier

class BehaviorMonitoringApp:
    def __init__(self):
        self.classifier = BehaviorClassifier()
        self.cap = None
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
    def draw_behavior_dashboard(self, frame, behavior_data, fps):
        """Draw behavior monitoring dashboard"""
        h, w = frame.shape[:2]
        
        behavior = behavior_data['behavior']
        confidence = behavior_data['confidence']
        risk_score = behavior_data['risk_score']
        hand_count = behavior_data.get('hand_count', 0)  # Safe access
        
        # Risk-based colors
        risk_colors = {
            0: (0, 255, 0),    # Green - safe
            1: (0, 255, 255),  # Yellow - low risk
            2: (0, 165, 255),  # Orange - medium risk  
            3: (0, 0, 255)     # Red - high risk
        }
        
        risk_color = risk_colors.get(risk_score, (0, 255, 0))
        
        # MAIN HEADER
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 80), risk_color, 3)
        
        # Behavior status
        behavior_text = f"BEHAVIOR: {behavior.upper()}"
        text_size = cv2.getTextSize(behavior_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, behavior_text, (text_x, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 2)
        
        # Risk indicator
        risk_text = f"RISK LEVEL: {risk_score}/3"
        cv2.putText(frame, risk_text, (text_x, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        # BEHAVIOR DESCRIPTION PANEL
        description = self.classifier.get_behavior_description(behavior)
        desc_size = cv2.getTextSize(description, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        desc_x = (w - desc_size[0]) // 2
        
        cv2.rectangle(frame, (desc_x-10, 90), (desc_x + desc_size[0] + 10, 120), 
                     (40, 40, 40), -1)
        cv2.putText(frame, description, (desc_x, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # LEFT PANEL - DETECTION INFO
        self._draw_detection_panel(frame, behavior_data, 20, 140, 300, 150)
        
        # RIGHT PANEL - RISK ANALYSIS
        self._draw_risk_panel(frame, behavior_data, w - 320, 140, 300, 150)
        
        # BOTTOM BAR - PERFORMANCE
        cv2.rectangle(frame, (0, h-40), (w, h), (40, 40, 40), -1)
        hands_status = "✅" if behavior_data['hands_detected'] else "❌"
        face_status = "✅" if behavior_data['face_detected'] else "❌"
        
        info_text = f"FPS: {fps:.1f} | Hands: {hands_status} | Face: {face_status} | Confidence: {confidence:.1f}"
        cv2.putText(frame, info_text, (20, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_detection_panel(self, frame, behavior_data, x, y, width, height):
        """Draw detection information panel"""
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        cv2.putText(frame, "DETECTION INFO", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        features = behavior_data['features']
        
        info_lines = [
            f"Motion: {features.get('motion_intensity', 0)}",
            f"Hand-like objects: {features.get('hand_like_objects', 0)}",
            f"Face active: {'Yes' if features.get('face_region_active', False) else 'No'}",
            f"Left hand active: {'Yes' if features.get('left_region_active', False) else 'No'}",
            f"Right hand active: {'Yes' if features.get('right_region_active', False) else 'No'}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (x + 10, y + 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_risk_panel(self, frame, behavior_data, x, y, width, height):
        """Draw risk analysis panel"""
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        cv2.putText(frame, "RISK ANALYSIS", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        risk_score = behavior_data['risk_score']
        
        # Risk meter
        meter_width = width - 40
        meter_x = x + 20
        meter_y = y + 50
        
        # Background
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + 20), 
                     (50, 50, 50), -1)
        
        # Progress
        progress_width = int((risk_score / 3) * meter_width)
        risk_color = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
        color = risk_color[risk_score]
        
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + progress_width, meter_y + 20), 
                     color, -1)
        
        # Recommendations
        recommendations = {
            0: "Continue safe driving",
            1: "Stay focused on road", 
            2: "Minimize distractions",
            3: "Immediate attention needed!"
        }
        
        cv2.putText(frame, f"Recommendation:", (meter_x, meter_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, recommendations[risk_score], (meter_x, meter_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def calculate_fps(self):
        """Calculate frames per second"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.fps
    
    def run(self):
        """Main behavior monitoring loop"""
        # Try to open camera with error handling
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Error: Could not open camera. Trying backup camera...")
                self.cap = cv2.VideoCapture(1)  # Try backup camera
                
            if not self.cap.isOpened():
                print("❌ Error: No cameras available")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("🚦 BEHAVIOR MONITORING SYSTEM STARTED")
            print("📋 Detecting:")
            print("   • Safe driving")
            print("   • Texting")
            print("   • Phone talking") 
            print("   • Drinking")
            print("   • Reaching behind")
            print("   • Operating radio")
            print("   • Press 'q' to quit")
            print("   • Green = Face region, Blue = Hand regions")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect behavior
                behavior_data = self.classifier.detect_behavior(frame)
                fps = self.calculate_fps()
                
                # Draw dashboard
                self.draw_behavior_dashboard(frame, behavior_data, fps)
                
                # Show frame
                cv2.imshow('Driver Behavior Monitoring', frame)
                
                # Handle exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ System shut down")

if __name__ == "__main__":
    app = BehaviorMonitoringApp()
    app.run()