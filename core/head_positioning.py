import cv2
import mediapipe as mp


class FacePositionAnalyzer:
    def __init__(self):
        """
        Initializes the Face Position Analyzer.
        """
        pass

    def analyze_positions(self, frame, face_landmarks_list):
        """
        Analyzes the position of faces in the frame and calculates bounding boxes and relative positions.
        :param frame: Input frame (BGR format).
        :param face_landmarks_list: List of face landmarks from FaceLandmarker.
        :return: List of dictionaries containing bounding box coordinates, percentages, and positions.
        """
        if not face_landmarks_list:
            return []

        h, w, _ = frame.shape
        frame_center_x, frame_center_y = w // 2, h // 2

        face_positions = []
        for face_landmarks in face_landmarks_list:
            cx_min, cy_min = w, h
            cx_max, cy_max = 0, 0

            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cx_min = min(cx_min, cx)
                cy_min = min(cy_min, cy)
                cx_max = max(cx_max, cx)
                cy_max = max(cy_max, cy)

            center_box_x = (cx_min + cx_max) // 2
            center_box_y = (cy_min + cy_max) // 2

            offset_x = center_box_x - frame_center_x
            offset_y = center_box_y - frame_center_y
            percentage_x = (offset_x / (w / 2)) * 100
            percentage_y = (offset_y / (h / 2)) * 100

            horizontal_position = "Right" if percentage_x > 0 else "Left"
            vertical_position = "Down" if percentage_y > 0 else "Up"

            face_positions.append({
                "bounding_box": (cx_min, cy_min, cx_max, cy_max),
                "center_box": (center_box_x, center_box_y),
                "percentage_x": percentage_x,
                "percentage_y": percentage_y,
                "horizontal_position": horizontal_position,
                "vertical_position": vertical_position,
            })

        return face_positions


def visualize(frame, face_positions):
    """
    Visualizes the face positions and bounding boxes on the frame.
    :param frame: Input frame (BGR format).
    :param face_positions: List of face position dictionaries from FacePositionAnalyzer.
    """
    h, w, _ = frame.shape
    frame_center_x, frame_center_y = w // 2, h // 2

    for face_data in face_positions:
        cx_min, cy_min, cx_max, cy_max = face_data["bounding_box"]
        center_box_x, center_box_y = face_data["center_box"]
        percentage_x = face_data["percentage_x"]
        percentage_y = face_data["percentage_y"]
        horizontal_position = face_data["horizontal_position"]
        vertical_position = face_data["vertical_position"]

        # Draw bounding box
        cv2.rectangle(frame, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)

        # Display the position and percentage information
        cv2.putText(frame, f"Horizontal: {horizontal_position} ({abs(percentage_x):.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Vertical: {vertical_position} ({abs(percentage_y):.1f}%)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw the center of the bounding box
        cv2.circle(frame, (center_box_x, center_box_y), 5, (0, 255, 0), -1)

    # Draw the center of the frame for reference
    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)
    cv2.putText(frame, "Frame Center (Origin)",
                (frame_center_x - 100, frame_center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    analyzer = FacePositionAnalyzer()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        face_positions = analyzer.analyze_positions(frame)
        visualize(frame, face_positions)

        cv2.imshow("Face Mesh with Relative Position", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
