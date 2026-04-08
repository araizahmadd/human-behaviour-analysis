import cv2
import mediapipe as mp
import numpy as np


class GazeIrisDetector:
    def __init__(self, eye_open_threshold=0.15, frames_threshold=10):
        """
        Initializes the Gaze and Iris Detector.
        :param eye_open_threshold: EAR threshold for eye open/closed detection.
        :param frames_threshold: Number of consecutive frames to confirm eyes closed.
        """
        self.eye_open_threshold = eye_open_threshold
        self.frames_threshold = frames_threshold
        self.closed_eye_count = 0
        self.gaze_history = []

        # Landmark indices
        self.LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS_LANDMARKS = [469, 470, 471, 472]
        self.RIGHT_IRIS_LANDMARKS = [474, 475, 476, 477]

    @staticmethod
    def calculate_ear(eye_landmarks):
        """
        Calculates the Eye Aspect Ratio (EAR).
        :param eye_landmarks: Landmarks of the eye.
        :return: EAR value.
        """
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    @staticmethod
    def calculate_iris_position(eye_landmarks, iris_landmarks):
        """
        Calculates the iris position relative to the eye.
        :param eye_landmarks: Eye landmarks.
        :param iris_landmarks: Iris landmarks.
        :return: Normalized iris position ratio.
        """
        eye_width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        iris_center = np.mean(iris_landmarks, axis=0)
        iris_to_left_corner = np.linalg.norm(iris_center - eye_landmarks[0])
        iris_position_ratio = iris_to_left_corner / eye_width
        return iris_position_ratio

    @staticmethod
    def calculate_orientation(landmarks):
        """
        Calculates head orientation (yaw, pitch, roll).
        :param landmarks: Face mesh landmarks.
        :return: Yaw, pitch, roll values.
        """
        nose_tip = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        chin = np.array([landmarks[152].x, landmarks[152].y, landmarks[152].z])
        left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])

        eye_vector = right_eye - left_eye
        face_vector = chin - nose_tip

        yaw = np.arctan2(eye_vector[2], eye_vector[0]) * 180 / np.pi
        pitch = np.arctan2(face_vector[2], face_vector[1]) * 180 / np.pi
        roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi

        return yaw, pitch, roll

    def compute_gaze(self, frame, face_landmarks_list):
        """
        Computes gaze status and iris position from the frame.
        :param frame: Input frame (BGR format).
        :param face_landmarks_list: List of face landmarks from FaceLandmarker.
        :return: Results containing status, gaze, yaw, pitch, roll, and landmark coordinates.
        """
        if not face_landmarks_list:
            return None

        h, w, _ = frame.shape
        for face_landmarks in face_landmarks_list:
            left_eye = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in self.LEFT_EYE_LANDMARKS])
            right_eye = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in self.RIGHT_EYE_LANDMARKS])

            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)

            if left_ear < self.eye_open_threshold and right_ear < self.eye_open_threshold:
                self.closed_eye_count += 1
                if self.closed_eye_count >= self.frames_threshold:
                    return {"status": "Eyes Closed"}
            else:
                self.closed_eye_count = 0

            left_iris = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in self.LEFT_IRIS_LANDMARKS])
            right_iris = np.array([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in self.RIGHT_IRIS_LANDMARKS])

            left_iris_position = self.calculate_iris_position(left_eye, left_iris)
            right_iris_position = self.calculate_iris_position(right_eye, right_iris)

            yaw, pitch, roll = self.calculate_orientation(face_landmarks)

            left_min, left_max = 0.35, 0.65
            right_min, right_max = 0.35, 0.60
            if left_min < left_iris_position < left_max and right_min < right_iris_position < right_max:
                self.gaze_history.append("Looking at Camera")
            else:
                self.gaze_history.append("Looking Away")

            if len(self.gaze_history) > 10:
                self.gaze_history.pop(0)

            final_gaze_status = "Looking Away" if self.gaze_history.count("Looking Away") >= 4 else "Looking at Camera"

            return {
                "status": "Eyes Open",
                "gaze_status": final_gaze_status,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "left_eye": left_eye,
                "right_eye": right_eye,
                "left_iris": left_iris,
                "right_iris": right_iris
            }


def visualize(frame, results):
    """
    Visualizes the gaze and iris detection results on the frame.
    :param frame: Input frame (BGR format).
    :param results: Detection results from GazeIrisDetector.
    """
    if not results:
        cv2.putText(frame, "No Face Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    status = results["status"]
    gaze_status = results.get("gaze_status", "")
    yaw = results.get("yaw", 0)
    left_eye = results.get("left_eye", [])
    right_eye = results.get("right_eye", [])
    left_iris = results.get("left_iris", [])
    right_iris = results.get("right_iris", [])

    cv2.putText(frame, f"Status: {status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if gaze_status:
        cv2.putText(frame, f"Gaze Status: {gaze_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for x, y in left_eye:
        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
    for x, y in right_eye:
        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
    for x, y in left_iris:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    for x, y in right_iris:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = GazeIrisDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.compute_gaze(frame)
        visualize(frame, results)

        cv2.imshow("Gaze and Iris Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
