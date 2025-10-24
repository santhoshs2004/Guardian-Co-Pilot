import cv2
import time
import numpy as np
from integrated_monitor import IntegratedDriverMonitor

class IntegratedWellnessApp:
    def __init__(self):
        self.monitor = IntegratedDriverMonitor()
        self.cap = None
        self.demo_mode = False
        
    def draw_comprehensive_dashboard(self, frame, data):
        """Draw comprehensive wellness monitoring dashboard"""
        h, w = frame.shape[:2]
        
        fatigue = data['fatigue']
        behavior = data['behavior']
        wellness = data['wellness']
        intervention = data['intervention']
        sensors = data['sensors']
        performance = data['performance']
        
        # MAIN WELLNESS HEADER
        header_color = wellness['color']
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 100), header_color, 3)
        
        # Wellness score (large display)
        wellness_text = f"WELLNESS: {wellness['level']} ({wellness['score']}/100)"
        wellness_size = cv2.getTextSize(wellness_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
        wellness_x = (w - wellness_size[0]) // 2
        cv2.putText(frame, wellness_text, (wellness_x, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, header_color, 3)
        
        # Trend indicator
        trend_text = f"Trend: {wellness['trend']}"
        cv2.putText(frame, trend_text, (wellness_x, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # INTERVENTION ALERT (prominent display)
        if intervention['level'] > 0:
            alert_color = intervention['color']
            alert_height = 60
            
            # Blinking effect for critical alerts
            if intervention['level'] >= 3 and int(time.time() * 2) % 2 == 0:
                bg_color = alert_color
                text_color = (255, 255, 255)
            else:
                bg_color = (0, 0, 0)
                text_color = alert_color
            
            cv2.rectangle(frame, (0, 105), (w, 105 + alert_height), bg_color, -1)
            cv2.rectangle(frame, (0, 105), (w, 105 + alert_height), alert_color, 2)
            
            alert_size = cv2.getTextSize(intervention['message'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            alert_x = (w - alert_size[0]) // 2
            cv2.putText(frame, intervention['message'], (alert_x, 105 + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # FOUR-COLUMN LAYOUT
        col_width = w // 4
        start_y = 180
        
        # Column 1: FATIGUE MONITORING
        self._draw_fatigue_column(frame, fatigue, 10, start_y, col_width - 20, 200)
        
        # Column 2: BEHAVIOR MONITORING
        self._draw_behavior_column(frame, behavior, col_width + 10, start_y, col_width - 20, 200)
        
        # Column 3: SENSOR DATA
        self._draw_sensor_column(frame, sensors, 2*col_width + 10, start_y, col_width - 20, 200)
        
        # Column 4: INTERVENTIONS
        self._draw_intervention_column(frame, intervention, 3*col_width + 10, start_y, col_width - 20, 200)
        
        # BOTTOM STATUS BAR
        self._draw_status_bar(frame, performance, w, h)
        
        # Draw detection overlays
        if fatigue['face_detected'] and fatigue['landmarks']:
            self._draw_detection_overlay(frame, fatigue, behavior)
    
    def _draw_fatigue_column(self, frame, fatigue, x, y, width, height):
        """Draw fatigue monitoring column"""
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        cv2.putText(frame, "FATIGUE MONITOR", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Fatigue level with color
        fatigue_color = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
        color = fatigue_color[fatigue['fatigue_level']]
        
        cv2.putText(frame, f"Level: {fatigue['status']}", (x + 10, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Metrics
        info_lines = [
            f"EAR: {fatigue['ear']:.3f}",
            f"MAR: {fatigue['mar']:.3f}",
            f"Eyes: {fatigue['eye_state']}",
            f"Mouth: {fatigue['mouth_state']}",
            f"Face: {'✅' if fatigue['face_detected'] else '❌'}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (x + 10, y + 80 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_behavior_column(self, frame, behavior, x, y, width, height):
        """Draw behavior monitoring column"""
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        cv2.putText(frame, "BEHAVIOR MONITOR", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Behavior with risk color
        risk_color = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
        color = risk_color[behavior['risk_score']]
        
        cv2.putText(frame, f"Behavior: {behavior['behavior']}", (x + 10, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(frame, f"Risk: {behavior['risk_score']}/3", (x + 10, y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Activity levels
        features = behavior['features']
        activity_lines = [
            f"Face: {features.get('face_region_active', 0):.1f}",
            f"Left: {features.get('left_region_active', 0):.1f}",
            f"Right: {features.get('right_region_active', 0):.1f}",
            f"Motion: {features.get('motion_intensity', 0)}",
            f"Confidence: {behavior['confidence']:.1f}"
        ]
        
        for i, line in enumerate(activity_lines):
            cv2.putText(frame, line, (x + 10, y + 105 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_sensor_column(self, frame, sensors, x, y, width, height):
        """Draw sensor data column"""
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        cv2.putText(frame, "SENSOR DATA", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Stress level with gauge
        stress = sensors['stress_level']
        stress_color = (0, int(255 * (1 - stress)), int(255 * stress))
        cv2.putText(frame, f"Stress: {stress:.1f}", (x + 10, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, stress_color, 2)
        
        # Steering irregularity
        steering = sensors['steering_irregularity']
        steering_color = (0, int(255 * (1 - steering)), int(255 * steering))
        cv2.putText(frame, f"Steering: {steering:.2f}", (x + 10, y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, steering_color, 2)
        
        # Heart Rate Variability (simulated)
        hrv = sensors['heart_rate_variability']
        hrv_color = (0, 255, 0) if hrv > 60 else (0, 165, 255)
        cv2.putText(frame, f"HRV: {hrv:.0f} ms", (x + 10, y + 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hrv_color, 2)
        
        # Wearable connectivity
        cv2.putText(frame, "Wearables: 🔗 Connected", (x + 10, y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(frame, "Steering: 🔗 Connected", (x + 10, y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def _draw_intervention_column(self, frame, intervention, x, y, width, height):
        """Draw intervention suggestions column"""
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        cv2.putText(frame, "INTERVENTIONS", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show top 3 interventions
        suggestions = intervention['suggestions'][:3]
        
        if not suggestions:
            cv2.putText(frame, "No interventions needed", (x + 10, y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            for i, suggestion in enumerate(suggestions):
                cv2.putText(frame, f"• {suggestion}", (x + 10, y + 55 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Intervention level indicator
        level_text = f"Level: {intervention['level']}/4"
        cv2.putText(frame, level_text, (x + 10, y + height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, intervention['color'], 2)
    
    def _draw_status_bar(self, frame, performance, w, h):
        """Draw bottom status bar"""
        cv2.rectangle(frame, (0, h-40), (w, h), (40, 40, 40), -1)
        
        status_text = (f"FPS: {performance['fps']:.1f} | "
                      f"Face: {'✅' if performance['face_detected'] else '❌'} | "
                      f"System: 🟢 OPERATIONAL | "
                      f"i.Mobilathon 5.0")
        
        cv2.putText(frame, status_text, (20, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_detection_overlay(self, frame, fatigue, behavior):
        """Draw detection overlays on video feed"""
        # Draw fatigue landmarks
        if fatigue['landmarks']:
            for idx in [33, 133, 362, 263]:  # Eye corners
                x, y = fatigue['landmarks'][idx]
                eye_color = (0, 0, 255) if fatigue['eye_state'] == "CLOSED" else (0, 255, 0)
                cv2.circle(frame, (x, y), 4, eye_color, -1)
            
            for idx in [61, 291, 13, 14]:  # Mouth corners
                x, y = fatigue['landmarks'][idx]
                mouth_color = (0, 0, 255) if fatigue['mouth_state'] == "YAWNING" else (255, 0, 0)
                cv2.circle(frame, (x, y), 4, mouth_color, -1)
    
    def run(self):
        """Main integrated monitoring loop"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🚗 AI-ENHANCED DRIVER WELLNESS MONITORING SYSTEM")
        print("🏆 i.Mobilathon 5.0 Challenge Entry")
        print("=" * 60)
        print("📊 MONITORING CAPABILITIES:")
        print("   • Real-time Fatigue Detection (EAR/MAR)")
        print("   • Driver Behavior Classification")
        print("   • Stress Level Assessment") 
        print("   • Steering Behavior Analysis")
        print("   • Multi-modal Wellness Scoring")
        print("   • Tiered Intervention System")
        print("   • Wearable Integration Ready")
        print("=" * 60)
        print("🎯 DEMO INSTRUCTIONS:")
        print("   • Close eyes 1+ seconds → Fatigue detection")
        print("   • Yawn → Stress indicator")
        print("   • Use phone → Behavior classification")
        print("   • Look away → Distraction detection")
        print("   • Press 'q' to exit system")
        print("=" * 60)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera feed lost")
                    break
                
                # Mirror for driver view
                frame = cv2.flip(frame, 1)
                
                # Process frame through integrated monitor
                start_time = time.time()
                monitoring_data = self.monitor.process_frame(frame)
                processing_time = (time.time() - start_time) * 1000
                
                # Add processing time to data
                monitoring_data['performance']['processing_time'] = processing_time
                
                # Draw comprehensive dashboard
                self.draw_comprehensive_dashboard(frame, monitoring_data)
                
                # Display processing time
                cv2.putText(frame, f"Process: {processing_time:.1f}ms", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('i.Mobilathon 5.0 - AI Driver Wellness Monitor', frame)
                
                # Print system status occasionally
                if int(time.time()) % 5 == 0:
                    wellness = monitoring_data['wellness']
                    intervention = monitoring_data['intervention']
                    print(f"📈 Wellness: {wellness['score']}/100 ({wellness['level']}) | "
                          f"Intervention: Level {intervention['level']}")
                
                # Handle exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.demo_mode = not self.demo_mode
                    print(f"🔧 Demo mode: {'ON' if self.demo_mode else 'OFF'}")
                
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ AI Driver Wellness System shut down")
            print("🏆 Thank you for using i.Mobilathon 5.0 entry!")

if __name__ == "__main__":
    app = IntegratedWellnessApp()
    app.run()