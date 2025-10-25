"""
Fixed FastAPI Backend with proper camera error handling
Place this in: Guardian-Co-Pilot/src/api.py
Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import asyncio
import threading
import queue
import time
import json
import uuid

# Import your existing detector
import cv2
from .fatigue_detector import FatigueDetector

# Initialize FastAPI app
app = FastAPI(
    title="Guardian Co-Pilot API",
    description="Real-time driver fatigue detection system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= GLOBAL STATE =============

detection_system = None
detection_running = False
current_session_id = None
metrics_queue = queue.Queue(maxsize=100)

# Data storage
sessions_db: Dict[str, List[Dict]] = {}
active_websockets: Dict[str, WebSocket] = {}

# ============= DATA MODELS =============

class SessionStartRequest(BaseModel):
    camera_id: int = 0
    session_name: Optional[str] = None

class CameraTestRequest(BaseModel):
    camera_id: int = 0

# ============= DETECTION THREAD =============

class BackendDetectionSystem:
    """Runs fatigue detection in background thread with proper error handling"""
    
    def __init__(self, session_id: str, camera_id: int = 0):
        self.session_id = session_id
        self.camera_id = camera_id
        self.detector = FatigueDetector()
        self.cap = None
        self.running = False
        self.thread = None
        self.camera_opened = False
        self.error_message = None
        self.frame_count = 0
        self.start_time = None
        
    def start(self):
        """Start detection in separate thread"""
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        
        # Wait up to 10 seconds for camera to open
        max_wait = 100  # 10 seconds
        for i in range(max_wait):
            if self.camera_opened:
                return True
            if self.error_message:
                return False
            time.sleep(0.1)
        
        # Timeout
        if not self.camera_opened:
            self.error_message = "Camera initialization timeout - try running camera_test.py to diagnose"
            self.stop()
            return False
        
        return True
        
    def _detection_loop(self):
        """Internal detection loop with robust error handling"""
        global detection_running
        
        try:
            print(f"📹 Opening camera {self.camera_id}...")
            
            # Try to open camera with multiple attempts
            for attempt in range(3):
                self.cap = cv2.VideoCapture(self.camera_id)
                
                if self.cap.isOpened():
                    break
                
                print(f"⚠️ Camera open attempt {attempt + 1} failed, retrying...")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(0.5)
            
            if not self.cap or not self.cap.isOpened():
                self.error_message = f"Failed to open camera {self.camera_id}"
                print(f"❌ {self.error_message}")
                return
            
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame read
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.error_message = "Camera opened but cannot read frames"
                print(f"❌ {self.error_message}")
                self.cap.release()
                self.cap = None
                return
            
            # Camera successfully opened
            self.camera_opened = True
            self.running = True
            self.start_time = time.time()
            detection_running = True
            
            print(f"✅ Camera {self.camera_id} opened successfully!")
            print(f"🚗 Detection started for session: {self.session_id}")
            print(f"📷 Camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
            frame_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while self.running and detection_running:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"⚠️ Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        print("❌ Too many consecutive frame read failures, stopping")
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Run detection
                try:
                    metrics = self.detector.detect_fatigue(frame)
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
                    time.sleep(0.1)
                    continue
                
                # Calculate FPS
                elapsed = time.time() - self.start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Package metrics
                metrics_data = {
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'ear': metrics['ear'],
                    'mar': metrics['mar'],
                    'eye_state': metrics['eye_state'],
                    'mouth_state': metrics['mouth_state'],
                    'fatigue_level': metrics['fatigue_level'],
                    'face_detected': metrics['face_detected'],
                    'status': metrics['status'],
                    'alert_message': metrics['alert_message'],
                    'fps': round(fps, 1),
                    'frame_count': frame_count
                }
                
                # Store metrics
                if self.session_id not in sessions_db:
                    sessions_db[self.session_id] = []
                sessions_db[self.session_id].append(metrics_data)
                
                # Put in queue for WebSocket
                try:
                    metrics_queue.put_nowait(metrics_data)
                except queue.Full:
                    try:
                        metrics_queue.get_nowait()  # Remove oldest
                        metrics_queue.put_nowait(metrics_data)
                    except:
                        pass
                
                # Check for critical alerts
                if metrics['fatigue_level'] >= 3:
                    print(f"🚨 CRITICAL ALERT - Session {self.session_id}: {metrics['status']}")
                
                # Control frame rate (~20-30 FPS)
                time.sleep(0.03)
                
        except Exception as e:
            self.error_message = str(e)
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop detection and cleanup"""
        global detection_running
        
        if not self.running and not detection_running:
            return  # Already stopped
        
        print(f"🛑 Stopping detection for session: {self.session_id}")
        self.running = False
        detection_running = False
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        print(f"✅ Detection stopped for session: {self.session_id}")

# ============= API ENDPOINTS =============

@app.get("/")
async def root():
    """Health check and API information"""
    return {
        "status": "online",
        "service": "Guardian Co-Pilot API",
        "version": "2.0.0",
        "detection_active": detection_running,
        "current_session": current_session_id,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "start_detection": "/api/detection/start",
            "stop_detection": "/api/detection/stop",
            "camera_test": "/api/camera/test",
            "list_cameras": "/api/camera/list"
        }
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    camera_status = "unknown"
    try:
        cap = cv2.VideoCapture(0)
        camera_status = "available" if cap.isOpened() else "unavailable"
        cap.release()
    except:
        camera_status = "error"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detection_running": detection_running,
        "active_sessions": len(sessions_db),
        "websocket_connections": len(active_websockets),
        "current_session": current_session_id,
        "camera_status": camera_status,
        "system_metrics": {
            "queue_size": metrics_queue.qsize(),
            "total_sessions": len(sessions_db)
        }
    }

@app.post("/api/detection/start")
async def start_detection(request: SessionStartRequest):
    """
    START DETECTION SYSTEM
    Returns error immediately if camera fails to open
    """
    global detection_system, detection_running, current_session_id
    
    # Force stop any existing session
    if detection_running or detection_system:
        print("⚠️ Force stopping existing session...")
        if detection_system:
            detection_system.stop()
            if detection_system.thread and detection_system.thread.is_alive():
                detection_system.thread.join(timeout=2)
            detection_system = None
        detection_running = False
        time.sleep(0.5)
    
    # Create new session
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if request.session_name:
        session_id = f"{request.session_name}_{session_id}"
    
    current_session_id = session_id
    sessions_db[session_id] = []
    
    print(f"📹 Attempting to open camera {request.camera_id} for session {session_id}...")
    
    # Start detection system
    detection_system = BackendDetectionSystem(session_id, request.camera_id)
    success = detection_system.start()
    
    if not success:
        error_msg = detection_system.error_message or f"Failed to start camera {request.camera_id}"
        error_msg += ". Please check: 1) Camera is connected, 2) No other app is using it, "
        error_msg += "3) Camera permissions are granted, 4) Try camera_id 0, 1, or 2"
        
        print(f"❌ {error_msg}")
        
        # Clean up failed attempt
        detection_system = None
        detection_running = False
        current_session_id = None
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    print(f"✅ Camera {request.camera_id} started successfully for session {session_id}!")
    
    return {
        "status": "started",
        "session_id": session_id,
        "camera_id": request.camera_id,
        "started_at": datetime.now().isoformat(),
        "message": "Detection system activated successfully",
        "websocket_url": f"wss://guardian-co-pilot-1.onrender.com/ws/{session_id}"
    }

@app.post("/api/detection/stop")
async def stop_detection():
    """STOP DETECTION SYSTEM"""
    global detection_system, detection_running, current_session_id
    
    if not detection_running and not detection_system:
        raise HTTPException(status_code=400, detail="Detection not running")
    
    session_id = current_session_id
    
    # Stop detection
    if detection_system:
        detection_system.stop()
        
        if detection_system.thread and detection_system.thread.is_alive():
            detection_system.thread.join(timeout=3)
        
        detection_system = None
    
    detection_running = False
    stopped_session = current_session_id
    current_session_id = None
    
    return {
        "status": "stopped",
        "session_id": stopped_session,
        "stopped_at": datetime.now().isoformat(),
        "message": "Detection system stopped successfully",
        "session_data_available": stopped_session in sessions_db
    }

@app.get("/api/camera/test")
async def test_camera(camera_id: int = 0):
    """Test if camera is available"""
    print(f"🧪 Testing camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        return {
            "available": False,
            "camera_id": camera_id,
            "message": f"Camera {camera_id} could not be opened",
            "suggestions": [
                "Check if camera is connected",
                "Close other apps using the camera (Zoom, Teams, etc.)",
                "Try different camera_id values (0, 1, 2)",
                "Check system camera permissions"
            ]
        }
    
    # Try to read a frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return {
            "available": False,
            "camera_id": camera_id,
            "message": f"Camera {camera_id} opened but cannot read frames",
            "suggestions": [
                "Camera might be in use by another application",
                "Try restarting the camera",
                "Check camera drivers"
            ]
        }
    
    return {
        "available": True,
        "camera_id": camera_id,
        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
        "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
        "message": f"Camera {camera_id} is working properly!",
        "details": {
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "data_type": str(frame.dtype)
        }
    }

@app.get("/api/camera/list")
async def list_cameras():
    """Find all available cameras"""
    print("🔍 Scanning for cameras...")
    available_cameras = []
    
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append({
                    "camera_id": i,
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                    "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                    "status": "working"
                })
            else:
                available_cameras.append({
                    "camera_id": i,
                    "status": "opened_but_no_frame"
                })
            cap.release()
        else:
            available_cameras.append({
                "camera_id": i,
                "status": "not_available"
            })
    
    working_cameras = [cam for cam in available_cameras if cam.get('status') == 'working']
    
    return {
        "available_cameras": available_cameras,
        "working_cameras": working_cameras,
        "count": len(working_cameras),
        "message": f"Found {len(working_cameras)} working camera(s) out of {len(available_cameras)} tested"
    }

@app.get("/api/detection/status")
async def detection_status():
    """Get current detection status"""
    return {
        "running": detection_running,
        "session_id": current_session_id,
        "timestamp": datetime.now().isoformat(),
        "details": {
            "camera_active": detection_system.camera_opened if detection_system else False,
            "frame_count": detection_system.frame_count if detection_system else 0
        }
    }

@app.post("/api/detection/reset")
async def reset_detection():
    """FORCE RESET - Stop any running detection"""
    global detection_system, detection_running, current_session_id
    
    print("🔄 Force resetting detection system...")
    
    if detection_system:
        detection_system.running = False
        if detection_system.cap:
            try:
                detection_system.cap.release()
            except:
                pass
        if detection_system.thread and detection_system.thread.is_alive():
            detection_system.thread.join(timeout=1)
        detection_system = None
    
    detection_running = False
    old_session = current_session_id
    current_session_id = None
    
    return {
        "status": "reset",
        "message": "Detection system has been force reset",
        "previous_session": old_session,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/metrics/latest")
async def get_latest_metrics():
    """Get most recent metrics"""
    if not current_session_id or current_session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="No active session")
    
    metrics = sessions_db[current_session_id]
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics available yet")
    
    return {
        "session_id": current_session_id,
        "latest_metrics": metrics[-1],
        "total_frames": len(metrics),
        "time_elapsed": f"{(datetime.now() - datetime.fromisoformat(metrics[0]['timestamp'])).total_seconds():.1f}s"
    }

@app.get("/api/session/{session_id}")
async def get_session(session_id: str, limit: int = 100):
    """Get recent metrics for session"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    metrics = sessions_db[session_id]
    return {
        "session_id": session_id,
        "total_records": len(metrics),
        "time_range": {
            "start": metrics[0]['timestamp'] if metrics else None,
            "end": metrics[-1]['timestamp'] if metrics else None
        },
        "recent_metrics": metrics[-limit:] if len(metrics) > limit else metrics
    }

@app.get("/api/analytics/{session_id}")
async def get_analytics(session_id: str):
    """Calculate analytics from session"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    metrics = sessions_db[session_id]
    
    if not metrics:
        return {"error": "No data available for this session"}
    
    # Calculate statistics
    total_frames = len(metrics)
    drowsy_events = sum(1 for m in metrics if m['fatigue_level'] >= 2)
    critical_events = sum(1 for m in metrics if m['fatigue_level'] >= 3)
    yawn_count = sum(1 for m in metrics if m['mouth_state'] == 'YAWNING')
    eyes_closed_count = sum(1 for m in metrics if m['eye_state'] == 'CLOSED')
    
    valid_metrics = [m for m in metrics if m['face_detected']]
    
    if valid_metrics:
        avg_ear = sum(m['ear'] for m in valid_metrics) / len(valid_metrics)
        avg_mar = sum(m['mar'] for m in valid_metrics) / len(valid_metrics)
        avg_fps = sum(m.get('fps', 0) for m in valid_metrics) / len(valid_metrics)
    else:
        avg_ear = avg_mar = avg_fps = 0
    
    # Time analysis
    first_time = datetime.fromisoformat(metrics[0]['timestamp'])
    last_time = datetime.fromisoformat(metrics[-1]['timestamp'])
    duration = (last_time - first_time).total_seconds()
    
    # Risk assessment
    risk_level = "LOW"
    if critical_events > 5:
        risk_level = "CRITICAL"
    elif critical_events > 2 or drowsy_events > 10:
        risk_level = "HIGH"
    elif drowsy_events > 5:
        risk_level = "MEDIUM"
    
    # Recommendation
    if critical_events > 5:
        recommendation = "🚨 URGENT: Stop driving immediately. Severe drowsiness detected."
    elif critical_events > 2:
        recommendation = "⚠️ WARNING: Take a break soon. Find a safe place to rest."
    elif drowsy_events > 10:
        recommendation = "⚠️ CAUTION: Consider taking a break. Multiple signs of fatigue."
    elif drowsy_events > 5:
        recommendation = "💤 MONITOR: Stay alert. Some signs of fatigue detected."
    else:
        recommendation = "✅ GOOD: Continue driving safely."
    
    # Fatigue timeline
    fatigue_timeline = []
    for i, m in enumerate(metrics):
        if m['fatigue_level'] >= 2:
            fatigue_timeline.append({
                "frame": i,
                "timestamp": m['timestamp'],
                "level": m['fatigue_level'],
                "status": m['status'],
                "ear": m['ear'],
                "mar": m['mar']
            })
    
    return {
        "session_id": session_id,
        "summary": {
            "duration_seconds": round(duration, 1),
            "duration_formatted": f"{int(duration // 60)}m {int(duration % 60)}s",
            "total_frames": total_frames,
            "drowsy_events": drowsy_events,
            "critical_events": critical_events,
            "yawn_count": yawn_count,
            "eyes_closed_count": eyes_closed_count,
            "average_ear": round(avg_ear, 3),
            "average_mar": round(avg_mar, 3),
            "average_fps": round(avg_fps, 1),
            "face_detection_rate": round(len(valid_metrics) / total_frames * 100, 1) if total_frames > 0 else 0
        },
        "fatigue_timeline": fatigue_timeline[-20:],  # Last 20 events
        "risk_assessment": {
            "level": risk_level,
            "recommendation": recommendation,
            "drowsiness_percentage": round(drowsy_events / total_frames * 100, 1) if total_frames > 0 else 0,
            "critical_percentage": round(critical_events / total_frames * 100, 1) if total_frames > 0 else 0
        }
    }

@app.get("/api/sessions/list")
async def list_sessions():
    """List all sessions"""
    sessions = []
    for session_id, metrics in sessions_db.items():
        if metrics:
            sessions.append({
                "session_id": session_id,
                "start_time": metrics[0]['timestamp'],
                "end_time": metrics[-1]['timestamp'],
                "frame_count": len(metrics),
                "last_update": metrics[-1]['timestamp'],
                "is_active": session_id == current_session_id and detection_running,
                "max_fatigue_level": max((m['fatigue_level'] for m in metrics), default=0)
            })
    
    return {
        "sessions": sorted(sessions, key=lambda x: x['start_time'], reverse=True),
        "total": len(sessions),
        "active_session": current_session_id
    }

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session data"""
    if session_id == current_session_id and detection_running:
        raise HTTPException(status_code=400, detail="Cannot delete active session")
    
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions_db[session_id]
    
    return {
        "status": "deleted",
        "session_id": session_id,
        "message": "Session data deleted successfully"
    }

# ============= WEBSOCKET =============

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Real-time WebSocket for live metrics"""
    await websocket.accept()
    active_websockets[session_id] = websocket
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to session {session_id}",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"🔌 WebSocket connected: {session_id}")
        
        last_heartbeat = time.time()
        last_metrics_time = time.time()
        
        while True:
            try:
                # Try to get metrics from queue
                try:
                    metrics = metrics_queue.get(timeout=1.0)
                    
                    if metrics['session_id'] == session_id:
                        await websocket.send_json({
                            "type": "metrics_update",
                            "data": metrics
                        })
                        last_metrics_time = time.time()
                    
                except queue.Empty:
                    # No new metrics, send heartbeat if needed
                    pass
                
                # Send heartbeat every 10 seconds if no metrics
                if time.time() - last_heartbeat > 10:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat(),
                        "session_active": session_id == current_session_id
                    })
                    last_heartbeat = time.time()
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"WebSocket inner error: {e}")
                break
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        if session_id in active_websockets:
            del active_websockets[session_id]
        print(f"🔌 WebSocket cleanup: {session_id}")

# ============= STARTUP/SHUTDOWN =============

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚗 GUARDIAN CO-PILOT API STARTED - Version 2.0.0")
    print("=" * 60)
    print("📡 Server: https://guardian-co-pilot-1.onrender.com")
    print("📖 API Docs: https://guardian-co-pilot-1.onrender.com/docs")
    print("🔌 WebSocket: wss://guardian-co-pilot-1.onrender.com/ws/{session_id}")
    print("🎥 Detection system ready")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    global detection_system, detection_running
    
    print("\n🛑 Shutting down Guardian Co-Pilot API...")
    
    # Close all WebSocket connections
    for session_id, websocket in list(active_websockets.items()):
        try:
            await websocket.close()
        except:
            pass
    active_websockets.clear()
    
    # Stop detection
    if detection_system:
        detection_system.stop()
    
    detection_running = False
    
    print("✅ Shutdown complete. Thank you for using Guardian Co-Pilot!")

# ============= MAIN =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )