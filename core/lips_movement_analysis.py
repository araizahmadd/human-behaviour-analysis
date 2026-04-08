import cv2
import mediapipe as mp
import numpy as np
from collections import deque


class LipMovementAnalyzer:
    def __init__(self, movement_threshold=1.5, speech_threshold=2, frame_history=15):
        """
        Initializes the Lip Movement Analyzer for speech detection.
        :param movement_threshold: Minimum change in lip distance to consider as speaking.
        :param speech_threshold: Minimum average movement for consistent speaking.
        :param frame_history: Number of frames to consider for movement history.
        """
        self.movement_threshold = movement_threshold
        self.speech_threshold = speech_threshold
        self.frame_history = frame_history
        self.lip_distances_history = deque(maxlen=frame_history)
        self.lip_distance_changes = deque(maxlen=frame_history)

        # Define mouth landmarks (outer and inner lips)
        self.MOUTH_LANDMARKS_OUTER = [61, 291, 0, 17]  # Outer lips
        self.MOUTH_LANDMARKS_INNER = [78, 308, 13, 14]  # Inner lips (top and bottom)

    @staticmethod
    def calculate_lip_distance(inner_lips):
        """
        Calculates the vertical distance between the inner lips.
        :param inner_lips: Coordinates of inner lips.
        :return: Lip height (distance).
        """
        lip_height = np.linalg.norm(inner_lips[0] - inner_lips[1])
        return lip_height

    def analyze_frame(self, frame, face_landmarks_list):
        """
        Analyzes a single frame for lip movement and speech status.
        :param frame: Input frame (BGR format).
        :param face_landmarks_list: List of face landmarks from FaceLandmarker.
        :return: Dictionary containing speech status, landmarks, and other metrics.
        """
        if not face_landmarks_list:
            return {"speech_status": "No Face Detected"}

        h, w, _ = frame.shape
        for face_landmarks in face_landmarks_list:
            # Get mouth landmarks
            outer_lips = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in self.MOUTH_LANDMARKS_OUTER])
            inner_lips = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in self.MOUTH_LANDMARKS_INNER])

            # Calculate lip distance
            lip_distance = self.calculate_lip_distance(inner_lips[:2])
            
            # Update movement history
            if self.lip_distances_history:
                lip_distance_change = abs(lip_distance - self.lip_distances_history[-1])
                self.lip_distance_changes.append(lip_distance_change)
            else:
                self.lip_distance_changes.append(0)

            self.lip_distances_history.append(lip_distance)

            # Calculate average movement
            avg_lip_change = np.mean(self.lip_distance_changes)

            # Determine speech status
            if avg_lip_change > self.movement_threshold and np.mean(self.lip_distances_history) > self.speech_threshold:
                speech_status = "Speaking"
            else:
                speech_status = "Silent"

            return {
                "speech_status": speech_status,
                "outer_lips": outer_lips,
                "inner_lips": inner_lips
            }


def visualize(frame, results):
    """
    Visualizes speech detection results and landmarks on the frame.
    :param frame: Input frame (BGR format).
    :param results: Detection results from LipMovementAnalyzer.
    """
    speech_status = results.get("speech_status", "No Face Detected")
    outer_lips = results.get("outer_lips", [])
    inner_lips = results.get("inner_lips", [])

    # Display speech status
    cv2.putText(frame, f"Speech Status: {speech_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Draw landmarks if available
    for x, y in outer_lips:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    for x, y in inner_lips:
        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    analyzer = LipMovementAnalyzer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = analyzer.analyze_frame(frame)
        visualize(frame, results)

        cv2.imshow("Speech Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
