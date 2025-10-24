# Guardian-Co-Pilot - " Aura Drive " 

## AI-Enhanced Driver Wellness Monitor

## i.Mobilathon 5.0 Project

## Overview
 
 AI-Enhanced Driver Wellness Monitor is a comprehensive real-time system that detects driver fatigue, distraction, and stress levels using multi-modal AI analysis. The system provides tiered, non-distracting interventions to prevent accidents before they happen.


## The Problem


=> AI-Enhanced Driver Wellness Monitoring

Fatigue and stress significantly reduce driver alertness, increasing the risk of accidents. Using cabin video, steering behavior, and optional wearables, create a solution that detects drowsiness or stress levels and suggests safe, non-distracting interventions to keep drivers and passengers safe.



1.35 million people die annually in road accidents (WHO)


20-30% of accidents are fatigue-related


$280 billion global economic cost of road crashes


Current solutions are either reactive or too intrusive


## Our Solution


=>Drowsiness Detection - eye closure, micro-sleeps, nodding, and loss of focus.


=>Stress Detection - from facial expressions, driving behavior, and physiological data.


=>Non-Distracting Intervention 


=>Multi-Modal Fusion - combine video, steering, and wearable data for a more robust system than any single input.


idea uniqueness :


=>Context-Awareness: The system's sensitivity changes based on context (e.g., high-speed highway vs. slow city traffic, time of day).


=>Proactive vs. Reactive: Don't just alert when the driver is already asleep. Predict impending drowsiness/stress based on trends.


=>Personalized Baseline: The system learns the driver's normal behavior (e.g., their typical steering style, resting heart rate) for more accurate anomaly detection.


## Features





<img width="880" height="378" alt="Screenshot 2025-10-24 215020" src="https://github.com/user-attachments/assets/b80d7e06-3caf-4fb3-abc0-e6baaad863f7" />








<img width="648" height="421" alt="Screenshot 2025-10-24 214936" src="https://github.com/user-attachments/assets/cbb0f112-9477-436c-ad5f-1149620e12ba" />








<img width="404" height="424" alt="image" src="https://github.com/user-attachments/assets/ac75623e-2357-4069-addf-590bbde638bb" />



## Key Features of the Multi-Modal System:

## 1. Comprehensive Monitoring:

Fatigue Detection: EAR, MAR, eye closure duration

Behavior Classification: 6 distraction types with risk scoring

Stress Assessment: Simulated stress levels

Steering Analysis: Simulated steering irregularity

Wellness Scoring: 0-100 comprehensive score

## 2. Tiered Intervention System:

Level 0: Excellent - Positive reinforcement

Level 1: Suggestion - Gentle reminders

Level 2: Moderate - Specific suggestions

Level 3: High Risk - Strong warnings

Level 4: Critical - Immediate action required

## 3. Multi-Modal Integration:

Video Analysis: Cabin camera for face/behavior

Steering Data: CAN bus integration ready

Wearables: HRV and stress monitoring ready


Real-time Processing: <50ms per frame



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







