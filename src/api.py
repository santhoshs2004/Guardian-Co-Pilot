"""
Guardian Co-Pilot API with Video Streaming Support
Backend receives video frames from frontend via WebSocket for processing
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import asyncio
import threading
import queue
import time
import json
import uuid
import base64
import cv2
import numpy as np
from .fatigue_detector import FatigueDetector

# Initialize FastAPI app
app = FastAPI(
    title="Guardian Co-Pilot API",
    description="Real-time driver fatigue detection system with video streaming",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= GLOBAL STATE =============

detection_system = None
detection_running = False
current_session_id = None
metrics_queue = queue.Queue(maxsize=100)
frame_processing_queue = queue.Queue(maxsize=10)  # Queue for frames from frontend

# Data storage
sessions_db: Dict[str, List[Dict]] = {}
active_websockets: Dict[str, WebSocket] = {}

# ============= DATA MODELS =============

class SessionStartRequest(BaseModel):
    camera_id: int = -1  # -1 indicates video streaming mode
    session_name: Optional[str] = None

# ============= VIDEO STREAMING DETECTION SYSTEM =============

class VideoStreamDetectionSystem:
    """Processes video frames received from frontend via WebSocket"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.detector = FatigueDetector()
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.start_time = None
        self.last_metrics_time = 0
        self.processed_frames = 0
        
    def start(self):
        """Start frame processing thread - NO CAMERA NEEDED"""
        print("🎮 Starting VIDEO STREAM processing mode - No local camera needed")
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        self.running = True
        self.start_time = time.time()
        return True
        
    def add_frame(self, frame_data: str, width: int, height: int):
        """Add a frame to the processing queue from frontend"""
        try:
            frame_processing_queue.put_nowait({
                'frame_data': frame_data,
                'width': width,
                'height': height,
                'timestamp': time.time()
            })
            return True
        except queue.Full:
            print("⚠️ Frame queue full, dropping frame")
            return False
    
    def _processing_loop(self):
        """Main processing loop for frames from frontend"""
        global detection_running
        
        try:
            print("🎮 Starting VIDEO STREAM processing mode")
            print("💡 Waiting for video frames from frontend...")
            
            self.running = True
            detection_running = True
            
            consecutive_empty_queues = 0
            max_consecutive_empty = 30  # 3 seconds at 10 FPS
            
            while self.running and detection_running:
                try:
                    # Get frame from queue with timeout
                    frame_info = frame_processing_queue.get(timeout=1.0)
                    
                    # Reset empty counter
                    consecutive_empty_queues = 0
                    
                    # Process the frame
                    self._process_frame(frame_info)
                    
                except queue.Empty:
                    consecutive_empty_queues += 1
                    if consecutive_empty_queues >= max_consecutive_empty:
                        print("⚠️ No frames received for a while, but keeping alive")
                        consecutive_empty_queues = max_consecutive_empty - 5
                    
                    # Send heartbeat metrics if no frames
                    if time.time() - self.last_metrics_time > 2.0:  # Every 2 seconds
                        self._send_heartbeat_metrics()
                    
                except Exception as e:
                    print(f"⚠️ Frame processing error: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Processing loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _process_frame(self, frame_info):
        """Process a single frame from frontend"""
        try:
            # Decode base64 frame data
            frame_data = base64.b64decode(frame_info['frame_data'])
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("⚠️ Failed to decode frame")
                return
            
            # Flip frame for mirror effect (optional)
            frame = cv2.flip(frame, 1)
            
            # Run fatigue detection
            metrics = self.detector.detect_fatigue(frame)
            
            # Calculate FPS
            self.frame_count += 1
            self.processed_frames += 1
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0
            
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
                'frame_count': self.processed_frames,
                'streaming_mode': True
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
                    metrics_queue.get_nowait()
                    metrics_queue.put_nowait(metrics_data)
                except:
                    pass
            
            # Update last metrics time
            self.last_metrics_time = time.time()
            
            # Log critical alerts
            if metrics['fatigue_level'] >= 3:
                print(f"🚨 CRITICAL ALERT (Streaming): {metrics['status']}")
            
            # Optional: Send processed frame back to frontend for display
            self._send_processed_frame(frame, metrics)
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
    
    def _send_processed_frame(self, frame, metrics):
        """Send processed frame back to frontend for display"""
        try:
            # Draw detection results on frame
            processed_frame = self._draw_detection_results(frame, metrics)
            
            # Encode frame to send back
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to frontend via WebSocket
            if self.session_id in active_websockets:
                websocket = active_websockets[self.session_id]
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({
                        "type": "processed_frame",
                        "frame_data": frame_base64,
                        "timestamp": datetime.now().isoformat()
                    }),
                    asyncio.get_event_loop()
                )
                
        except Exception as e:
            print(f"⚠️ Error sending processed frame: {e}")
    
    def _draw_detection_results(self, frame, metrics):
        """Draw detection results on frame for visualization"""
        try:
            # Create a copy of the frame
            display_frame = frame.copy()
            
            # Get frame dimensions
            height, width = display_frame.shape[:2]
            
            # Draw status text
            status_text = f"Status: {metrics['status']}"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw metrics
            ear_text = f"EAR: {metrics['ear']:.3f}"
            mar_text = f"MAR: {metrics['mar']:.3f}"
            fatigue_text = f"Fatigue: {metrics['fatigue_level']}"
            
            cv2.putText(display_frame, ear_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, mar_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, fatigue_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Color code based on fatigue level
            if metrics['fatigue_level'] == 0:
                color = (0, 255, 0)  # Green
            elif metrics['fatigue_level'] == 1:
                color = (255, 255, 0)  # Yellow
            elif metrics['fatigue_level'] == 2:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw border based on fatigue level
            cv2.rectangle(display_frame, (0, 0), (width-1, height-1), color, 3)
            
            return display_frame
            
        except Exception as e:
            print(f"⚠️ Error drawing results: {e}")
            return frame
    
    def _send_heartbeat_metrics(self):
        """Send heartbeat metrics when no frames are being processed"""
        try:
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0
            
            heartbeat_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'ear': 0.0,
                'mar': 0.0,
                'eye_state': 'UNKNOWN',
                'mouth_state': 'UNKNOWN',
                'fatigue_level': 0,
                'face_detected': False,
                'status': 'Waiting for video frames...',
                'alert_message': None,
                'fps': round(fps, 1),
                'frame_count': self.processed_frames,
                'streaming_mode': True,
                'heartbeat': True
            }
            
            try:
                metrics_queue.put_nowait(heartbeat_data)
            except queue.Full:
                pass
                
            self.last_metrics_time = time.time()
            
        except Exception as e:
            print(f"⚠️ Heartbeat error: {e}")
    
    def stop(self):
        """Stop processing and cleanup"""
        global detection_running
        
        if not self.running:
            return
        
        print(f"🛑 Stopping video stream processing for session: {self.session_id}")
        self.running = False
        detection_running = False
        
        # Clear frame queue
        while not frame_processing_queue.empty():
            try:
                frame_processing_queue.get_nowait()
            except:
                break
        
        print(f"✅ Video stream processing stopped for session: {self.session_id}")

# ============= API ENDPOINTS =============

@app.get("/")
async def root():
    """Health check and API information"""
    return {
        "status": "online",
        "service": "Guardian Co-Pilot API",
        "version": "3.0.0",
        "detection_active": detection_running,
        "current_session": current_session_id,
        "mode": "video_streaming",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "start_detection": "/api/detection/start",
            "stop_detection": "/api/detection/stop"
        }
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detection_running": detection_running,
        "active_sessions": len(sessions_db),
        "websocket_connections": len(active_websockets),
        "current_session": current_session_id,
        "mode": "video_streaming",
        "queue_status": {
            "metrics_queue": metrics_queue.qsize(),
            "frame_queue": frame_processing_queue.qsize()
        },
        "system_metrics": {
            "total_sessions": len(sessions_db),
            "total_frames_processed": detection_system.processed_frames if detection_system else 0
        }
    }

@app.post("/api/detection/start")
async def start_detection(request: SessionStartRequest):
    """
    START DETECTION SYSTEM - Video Streaming Mode
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
    session_id = f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if request.session_name:
        session_id = f"{request.session_name}_{session_id}"
    
    current_session_id = session_id
    sessions_db[session_id] = []
    
    print(f"🚀 Starting VIDEO STREAM detection for session: {session_id}")
    
    # Start video stream detection system
    detection_system = VideoStreamDetectionSystem(session_id)
    success = detection_system.start()
    
    if not success:
        error_msg = "Failed to start video stream detection system"
        print(f"❌ {error_msg}")
        
        detection_system = None
        detection_running = False
        current_session_id = None
        
        raise HTTPException(status_code=500, detail=error_msg)
    
    print(f"✅ Video stream detection started successfully!")
    print("💡 Waiting for video frames from frontend...")
    
    return {
        "status": "started",
        "session_id": session_id,
        "mode": "video_streaming",
        "started_at": datetime.now().isoformat(),
        "message": "Video stream detection activated. Send video frames via WebSocket.",
        "websocket_url": f"wss://guardian-co-pilot-1.onrender.com/ws/{session_id}",
        "instructions": {
            "frame_format": "base64 JPEG",
            "frame_rate": "Recommended: 5-10 FPS",
            "resolution": "Recommended: 640x480",
            "message_type": "Send 'video_frame' messages via WebSocket"
        }
    }

@app.post("/api/detection/stop")
async def stop_detection():
    """STOP DETECTION SYSTEM"""
    global detection_system, detection_running, current_session_id
    
    if not detection_running and not detection_system:
        raise HTTPException(status_code=400, detail="Detection not running")
    
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
        "message": "Video stream detection stopped successfully",
        "session_data_available": stopped_session in sessions_db,
        "total_frames_processed": sessions_db[stopped_session][-1]['frame_count'] if stopped_session in sessions_db and sessions_db[stopped_session] else 0
    }

# ============= WEBSOCKET FOR VIDEO STREAMING =============

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Real-time WebSocket for video frames and metrics"""
    await websocket.accept()
    active_websockets[session_id] = websocket
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to video stream session {session_id}",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "instructions": {
                "send_frames": "Send 'video_frame' messages with base64 JPEG data",
                "frame_rate": "Recommended: 5-10 FPS",
                "format": "Include width, height, and frame_data"
            }
        })
        
        print(f"🔌 WebSocket connected for video streaming: {session_id}")
        
        last_heartbeat = time.time()
        frames_received = 0
        
        while True:
            try:
                # Receive message from frontend
                data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                message = json.loads(data)
                
                if message['type'] == 'video_frame':
                    # Process video frame from frontend
                    frames_received += 1
                    
                    if detection_system and session_id == current_session_id:
                        success = detection_system.add_frame(
                            frame_data=message['frame_data'],
                            width=message.get('width', 640),
                            height=message.get('height', 480)
                        )
                        
                        if success and frames_received % 30 == 0:  # Every 30 frames
                            print(f"📹 Frames received: {frames_received}")
                    
                    last_heartbeat = time.time()
                    
                elif message['type'] == 'ping':
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except asyncio.TimeoutError:
                # Send heartbeat if no activity
                if time.time() - last_heartbeat > 10:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat(),
                        "session_active": session_id == current_session_id,
                        "frames_received": frames_received
                    })
                    last_heartbeat = time.time()
                    
            except Exception as e:
                print(f"WebSocket receive error: {e}")
                break
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        if session_id in active_websockets:
            del active_websockets[session_id]
        print(f"🔌 WebSocket cleanup: {session_id} - Frames received: {frames_received}")

# ============= METRICS WEBSOCKET HANDLER =============

async def metrics_broadcaster():
    """Broadcast metrics to all connected WebSocket clients"""
    while True:
        try:
            # Get metrics from queue
            metrics = metrics_queue.get(timeout=1.0)
            session_id = metrics['session_id']
            
            # Send to appropriate WebSocket client
            if session_id in active_websockets:
                websocket = active_websockets[session_id]
                try:
                    await websocket.send_json({
                        "type": "metrics_update",
                        "data": metrics
                    })
                except:
                    # Remove disconnected client
                    if session_id in active_websockets:
                        del active_websockets[session_id]
                        
        except queue.Empty:
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Metrics broadcaster error: {e}")
            await asyncio.sleep(1.0)

# ============= STARTUP/SHUTDOWN =============

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚗 GUARDIAN CO-PILOT API STARTED - Version 3.0.0")
    print("=" * 60)
    print("📡 Server: https://guardian-co-pilot-1.onrender.com")
    print("📖 API Docs: https://guardian-co-pilot-1.onrender.com/docs")
    print("🔌 WebSocket: wss://guardian-co-pilot-1.onrender.com/ws/{session_id}")
    print("🎥 VIDEO STREAMING MODE: Ready to receive frames from frontend")
    print("=" * 60)
    
    # Start metrics broadcaster
    asyncio.create_task(metrics_broadcaster())

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