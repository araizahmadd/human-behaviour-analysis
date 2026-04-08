import cv2
import time
import mediapipe as mp
import numpy as np

# Define function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    # Vertical distances
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def get_ear(face_landmarks):
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # Left eye: [left_corner, top_1, top_2, right_corner, bottom_1, bottom_2]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # Right eye: [left_corner, top_1, top_2, right_corner, bottom_1, bottom_2]

    # Get eye landmarks
    left_eye = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in LEFT_EYE_LANDMARKS])
    right_eye = np.array([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in RIGHT_EYE_LANDMARKS])

    # Calculate EAR for both eyes
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)

    return left_ear, right_ear, left_eye, right_eye

if __name__ == '__main__':
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    # Open webcam
    cap = cv2.VideoCapture(0)

    closed_eye_count = 0
    frames_threshold = 10



    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
               
                left_ear, right_ear, left_eye, right_eye = get_ear(face_landmarks)
                # Threshold to determine if eyes are open or closed
                eye_open_threshold = 0.2
                if left_ear < eye_open_threshold and right_ear < eye_open_threshold:
                    closed_eye_count += 1
                    if closed_eye_count >= frames_threshold:
                        status = "Eyes Closed"
                    else:
                        status = "Eyes Open"
                else:
                    status = "Eyes Open"
                    closed_eye_count = 0

                # Display the EAR values and status
                cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, f"Status: {status}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw the eye landmarks
                for x, y in left_eye:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                for x, y in right_eye:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

        # Show the frame
        cv2.imshow("Eye Open/Closed Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

# Release resources
cap.release()
cv2.destroyAllWindows()