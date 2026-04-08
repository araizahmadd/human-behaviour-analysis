import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Add parent directory to path to resolve 'core' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.gaze_estimation import GazeIrisDetector
from core.head_positioning import FacePositionAnalyzer
from core.lips_movement_analysis import LipMovementAnalyzer
from core.distance_estimation import DistanceEstimator
from core.main import visualize_distance, visualize_gaze, visualize_head, visualize_lip_movement

app = FastAPI(title="Human Face Analysis Video Stream API")

# Setup CORS to allow requests from our Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
gaze_detector = GazeIrisDetector()
position_analyzer = FacePositionAnalyzer()
lip_analyzer = LipMovementAnalyzer()
distance_estimator = DistanceEstimator()

# Initialize MediaPipe Tasks Vision FaceLandmarker
base_options = python.BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), '..', 'core', 'face_landmarker.task'))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

def generate_frames():
    """Generator function that grabs video frames, processes them, and yields MJPEG chunks."""
    import numpy as np
    import time
    
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    # Check if the camera was successfully opened. If not, it means permission was denied or camera is missing.
    if not cap.isOpened() or not cap.read()[0]:
        print("\n\n--- ERROR: CANNOT ACCESS WEBCAM ---")
        print("Please enable Camera permissions for your Terminal in Mac Settings -> Privacy & Security -> Camera\n\n")
        
        # Yield an error frame so the frontend doesn't just show a broken image link
        while True:
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "Camera Access Denied or Not Found!", (30, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(err_frame, "1. Open Mac System Settings", (30, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(err_frame, "2. Go to Privacy & Security -> Camera", (30, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(err_frame, "3. Enable permission for your Terminal", (30, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw a pulsing red dot to show it's still streaming the error
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(err_frame, (580, 50), 10, (0, 0, 255), -1)

            ret, buffer = cv2.imencode('.jpg', err_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)

        # Process frame with MediaPipe Tasks API FaceLandmarker
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)
        
        face_landmarks_list = detection_result.face_landmarks

        # Apply computer vision models
        gaze_results = gaze_detector.compute_gaze(frame, face_landmarks_list)
        position_results = position_analyzer.analyze_positions(frame, face_landmarks_list)
        lip_results = lip_analyzer.analyze_frame(frame, face_landmarks_list)
        distance_results = distance_estimator.compute_distance(frame, face_landmarks_list)
        
        # Apply visual overlays
        visualize_gaze(frame, gaze_results, offset_y=30)
        visualize_head(frame, position_results, offset_y=90)
        visualize_lip_movement(frame, lip_results, offset_y=150)
        visualize_distance(frame, distance_results)
        
        # Encode the modified frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.get("/video_feed")
def video_feed():
    """Streams the real-time processed video."""
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    # Can be run via `python -m api.main` or `uvicorn api.main:app --reload`
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
