"""
Simple test client to control your detection system via API
Place this in: Guardian-Co-Pilot/src/test_api.py
Run with: python test_api.py
"""

import requests
import time
import json

API_URL = "http://localhost:8000"

def print_response(response):
    """Pretty print API response"""
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except:
        print(response.text)

def check_health():
    """Check if backend is running"""
    print("\n🔍 Checking backend health...")
    response = requests.get(f"{API_URL}/api/health")
    print_response(response)
    return response.status_code == 200

def start_detection():
    """Start detection system"""
    print("\n▶️  Starting detection system...")
    response = requests.post(f"{API_URL}/api/detection/start", json={"camera_id": 0})
    print_response(response)
    
    if response.status_code == 200:
        data = response.json()
        return data.get('session_id')
    return None

def stop_detection():
    """Stop detection system"""
    print("\n⏸️  Stopping detection system...")
    response = requests.post(f"{API_URL}/api/detection/stop")
    print_response(response)

def get_status():
    """Get detection status"""
    print("\n📊 Detection status:")
    response = requests.get(f"{API_URL}/api/detection/status")
    print_response(response)

def get_latest_metrics():
    """Get latest metrics"""
    print("\n📈 Latest metrics:")
    response = requests.get(f"{API_URL}/api/metrics/latest")
    print_response(response)

def get_analytics(session_id):
    """Get analytics for session"""
    print(f"\n📊 Analytics for session {session_id}:")
    response = requests.get(f"{API_URL}/api/analytics/{session_id}")
    print_response(response)

def list_sessions():
    """List all sessions"""
    print("\n📋 All sessions:")
    response = requests.get(f"{API_URL}/api/sessions/list")
    print_response(response)

def interactive_menu():
    """Interactive menu"""
    session_id = None
    
    while True:
        print("\n" + "="*50)
        print("🚗 GUARDIAN CO-PILOT - API CONTROLLER")
        print("="*50)
        print("1. Check Backend Health")
        print("2. Start Detection")
        print("3. Stop Detection")
        print("4. Get Detection Status")
        print("5. Get Latest Metrics")
        print("6. Get Analytics (requires session ID)")
        print("7. List All Sessions")
        print("8. Run Detection for 30 seconds")
        print("9. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            check_health()
        
        elif choice == '2':
            session_id = start_detection()
            if session_id:
                print(f"\n✅ Detection started! Session ID: {session_id}")
        
        elif choice == '3':
            stop_detection()
        
        elif choice == '4':
            get_status()
        
        elif choice == '5':
            try:
                get_latest_metrics()
            except:
                print("❌ No active session or no metrics yet")
        
        elif choice == '6':
            if not session_id:
                sid = input("Enter session ID: ").strip()
            else:
                sid = session_id
            get_analytics(sid)
        
        elif choice == '7':
            list_sessions()
        
        elif choice == '8':
            print("\n🎬 Starting 30-second detection test...")
            session_id = start_detection()
            
            if session_id:
                print(f"✅ Session started: {session_id}")
                print("⏳ Running for 30 seconds...")
                
                # Monitor for 30 seconds
                for i in range(30):
                    time.sleep(1)
                    if (i + 1) % 5 == 0:
                        print(f"⏰ {i + 1} seconds elapsed...")
                
                print("\n📊 Getting final analytics...")
                get_analytics(session_id)
                
                print("\n🛑 Stopping detection...")
                stop_detection()
        
        elif choice == '9':
            print("\n👋 Goodbye!")
            # Stop detection if running
            try:
                stop_detection()
            except:
                pass
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🚀 Guardian Co-Pilot API Test Client")
    print("=" * 50)
    
    # Check if backend is running
    try:
        if not check_health():
            print("\n❌ Backend is not running!")
            print("Start it with: uvicorn api:app --reload")
            exit(1)
        
        print("\n✅ Backend is running!")
        
        # Start interactive menu
        interactive_menu()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to backend!")
        print("Make sure the backend is running:")
        print("  cd Guardian-Co-Pilot/src")
        print("  uvicorn api:app --reload")
        exit(1)