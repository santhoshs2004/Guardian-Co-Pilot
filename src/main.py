import cv2
import time
import numpy as np
from fatigue_detector import FatigueDetector
#from fatigue_detector import SimpleFatigueDetector as FatigueDetector

class CarFatigueDetectionApp:
    def __init__(self):
        self.detector = FatigueDetector()
        self.cap = None
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
    def draw_dashboard(self, frame, metrics, fps):
        """Draw enhanced dashboard with better visualization"""
        h, w = frame.shape[:2]
        
        # Colors based on fatigue level
        color_map = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
        status_color = color_map[metrics['fatigue_level']]
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 70), status_color, 2)
        
        # Large status text
        status_text = metrics['status']
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, status_text, (text_x, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Face detection status
        face_status = "FACE DETECTED ✅" if metrics['face_detected'] else "NO FACE ❌"
        cv2.putText(frame, face_status, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Alert message
        if metrics['alert_message']:
            # Blinking effect for severe alerts
            if metrics['fatigue_level'] == 3 and int(time.time() * 3) % 2 == 0:
                bg_color = (0, 0, 255)
                text_color = (255, 255, 255)
            else:
                bg_color = (0, 0, 0)
                text_color = status_color
            
            alert_size = cv2.getTextSize(metrics['alert_message'], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            alert_x = (w - alert_size[0]) // 2
            
            cv2.rectangle(frame, (alert_x-15, 75), (alert_x + alert_size[0] + 15, 115), 
                         bg_color, -1)
            cv2.putText(frame, metrics['alert_message'], (alert_x, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Metrics panel
        metrics_bg = (40, 40, 40)
        cv2.rectangle(frame, (10, h-120), (300, h-10), metrics_bg, -1)
        cv2.rectangle(frame, (10, h-120), (300, h-10), (100, 100, 100), 2)
        
        cv2.putText(frame, f"EAR: {metrics['ear']:.3f}", (20, h-90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {metrics['mar']:.3f}", (20, h-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, h-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Eye state indicator
        eye_color = (0, 0, 255) if metrics['eye_state'] == "CLOSED" else (0, 255, 0)
        cv2.putText(frame, f"Eyes: {metrics['eye_state']}", (150, h-90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        
        # Mouth state indicator  
        mouth_color = (0, 0, 255) if metrics['mouth_state'] == "YAWNING" else (255, 0, 0)
        cv2.putText(frame, f"Mouth: {metrics['mouth_state']}", (150, h-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
        
        # Draw landmarks if face detected
        if metrics['face_detected'] and metrics['landmarks']:
            # Draw key points
            for idx in [33, 133, 362, 263]:  # Eye corners
                x, y = metrics['landmarks'][idx]
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            for idx in [61, 291, 13, 14]:  # Mouth corners
                x, y = metrics['landmarks'][idx]
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    
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
        """Main application loop"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🚗 FATIGUE DETECTION SYSTEM STARTED")
        print("📋 Instructions:")
        print("   • Ensure good lighting")
        print("   • Face the camera directly")
        print("   • Close eyes for 1+ seconds to test")
        print("   • Yawn to test mouth detection")
        print("   • Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Detect fatigue
                metrics = self.detector.detect_fatigue(frame)
                fps = self.calculate_fps()
                
                # Draw interface
                self.draw_dashboard(frame, metrics, fps)
                
                # Show frame
                cv2.imshow('Driver Fatigue Detection', frame)
                
                # Handle exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n🛑 System stopped")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ System shut down")

if __name__ == "__main__":
    app = CarFatigueDetectionApp()
    app.run()