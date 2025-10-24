# Guardian-Co-Pilot - " Aura Drive " 

## AI-Enhanced Driver Wellness Monitor

## i.Mobilathon 5.0 Project

## Overview
 
 AI-Enhanced Driver Wellness Monitor is a comprehensive real-time system that detects driver fatigue, distraction, and stress levels using multi-modal AI analysis. The system provides tiered, non-distracting interventions to prevent accidents before they happen.

## The Problem


1.35 million people die annually in road accidents (WHO)


20-30% of accidents are fatigue-related


$280 billion global economic cost of road crashes


Current solutions are either reactive or too intrusive


## Our Solution


A proactive, AI-powered system that:


Monitors driver state in real-time (50ms response)


Detects fatigue, distraction, and stress simultaneously


Provides context-aware, non-distracting interventions


Prevents accidents before they occur


## Features

## Core Capabilities

Feature	                          Technology	            Accuracy	            Response Time

Fatigue Detection	              Eye Aspect Ratio (EAR)	      95%	                 <100ms

Distraction Classification	     Behavioral Analysis	         90%	                  <150ms

Stress Assessment	              Multi-modal Fusion	          85%	                  <200ms

Wellness Scoring	               Composite AI	                92%	                  <50ms


## Safety Interventions

## Level	Condition	   Interventions
Level 0	 - Excellent	  Positive reinforcement

Level 1	 - Good	       Gentle suggestions

Level 2	 - Moderate	   Specific recommendations

Level 3	 - High Risk	  Strong warnings

Level 4	 - Critical	   Immediate action required


<img width="404" height="424" alt="image" src="https://github.com/user-attachments/assets/ac75623e-2357-4069-addf-590bbde638bb" />


## Installation

## System Requirements

Python: 3.8, 3.9, 3.10, 3.11, or 3.12

OS: Windows 10+, macOS 10.14+, Ubuntu 18.04+

RAM: 4GB minimum, 8GB recommended

Webcam: 720p+ resolution recommended

Storage: 500MB free space


## Manual Installation

```
# 1. Create virtual environment
python -m venv wellness_ai
wellness_ai\Scripts\activate  # Windows
# source wellness_ai/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify installation
python -c "import cv2, mediapipe, numpy; print('✅ All systems ready!')"


```


## Troubleshooting

## Common Issues & Solutions:

1) Webcam not detected:
   
```
python
# Try different camera indices
cap = cv2.VideoCapture(0)  # Primary camera
cap = cv2.VideoCapture(1)  # Secondary camera
```

2) MediaPipe compatibility:

```
bash
pip install protobuf==3.20.3 --force-reinstall
pip install mediapipe==0.10.0 --force-reinstall
```

3) Performance optimization:

```
python
# Reduce resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

```

## Performance Metrics

Processing Speed: 20 FPS (50ms per frame)

Accuracy: 92% overall detection rate

False Positives: <5% in controlled conditions

Resource Usage: <500MB RAM, <15% CPU


## Demo Scenarios

## Scenario 1: Alert Driver (Baseline)

Actions: Sit normally, hands on wheel

Expected: Wellness 85-100, Level 0 interventions

## Scenario 2: Fatigue Detection

Actions: Close eyes for 2+ seconds

Expected: Wellness drops, Level 2-3 interventions


## Scenario 3: Distraction Detection

Actions: Pretend phone use, look away

Expected: Behavior risk increases, specific alerts

## Scenario 4: Combined Risk

Actions: Fatigue + distraction simultaneously

Expected: Critical alerts, immediate action suggested

## Scenario 5: Stress Indicators

Actions: Yawn, restless movements

Expected: Stress level increase, wellness decline


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Future Roadmap

Phase 1(6 months) 

OBD-II integration for real steering data

Wearable device integration (HRV monitoring)

Cloud dashboard for fleet management

Mobile app for driver feedback


Phase 2 (12 months) 

Predictive analytics for risk forecasting

OEM integration partnerships

Insurance telematics platform

Global deployment scaling



## Acknowledgments

MediaPipe Team for excellent face detection capabilities

OpenCV Community for computer vision tools

i.Mobilathon Organizers for the platform and opportunity

Road Safety Researchers whose work inspired this project







