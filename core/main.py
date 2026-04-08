import cv2
from .gaze_estimation import GazeIrisDetector
from .head_positioning import FacePositionAnalyzer
from .lips_movement_analysis import LipMovementAnalyzer
from .distance_estimation import DistanceEstimator


def visualize_distance(frame, results):
    """
    Visualizes the distance estimation results on the frame.
    :param frame: Input frame (BGR image).
    :param results: List of tuples containing distance, distance category, and key coordinates.
    """
    for distance, category, foreheadcoords, chincoords in results:
        cv2.circle(frame, foreheadcoords, 5, (0, 255, 0), -1)
        cv2.circle(frame, chincoords, 5, (0, 0, 255), -1)
        cv2.putText(frame, f'Distance: {int(distance)}',
                    (foreheadcoords[0] + 10, foreheadcoords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f'Category: {category}',
                    (foreheadcoords[0] + 10, foreheadcoords[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def visualize_gaze(frame, results, offset_y=30):
    """
    Visualizes the gaze and iris detection results on the frame.
    :param frame: Input frame (BGR format).
    :param results: Detection results from GazeIrisDetector.
    :param offset_y: Y-offset for the text placement.
    """
    if not results:
        cv2.putText(frame, "No Face Detected", (30, offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    status = results["status"]
    gaze_status = results.get("gaze_status", "")

    cv2.putText(frame, f"Gaze Status: {gaze_status}", (30, offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Eye Status: {status}", (30, offset_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def visualize_head(frame, face_positions, offset_y=90):
    """
    Visualizes the face positions and bounding boxes on the frame.
    :param frame: Input frame (BGR format).
    :param face_positions: List of face position dictionaries from FacePositionAnalyzer.
    :param offset_y: Y-offset for the text placement.
    """
    for face_data in face_positions:
        percentage_x = face_data["percentage_x"]
        percentage_y = face_data["percentage_y"]
        horizontal_position = face_data["horizontal_position"]
        vertical_position = face_data["vertical_position"]

        # Display the position and percentage information
        cv2.putText(frame, f"Horizontal: {horizontal_position} ({abs(percentage_x):.1f}%)",
                    (30, offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Vertical: {vertical_position} ({abs(percentage_y):.1f}%)",
                    (30, offset_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def visualize_lip_movement(frame, results, offset_y=150):
    """
    Visualizes speech detection results and landmarks on the frame.
    :param frame: Input frame (BGR format).
    :param results: Detection results from LipMovementAnalyzer.
    :param offset_y: Y-offset for the text placement.
    """
    speech_status = results.get("speech_status", "No Face Detected")

    # Display speech status
    cv2.putText(frame, f"Speech Status: {speech_status}", (30, offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def main():
    # Initialize models for all modules
    gaze_detector = GazeIrisDetector()
    position_analyzer = FacePositionAnalyzer()
    lip_analyzer = LipMovementAnalyzer()
    distance_estimator = DistanceEstimator()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame for better user experience

        # Process each module sequentially
        gaze_results = gaze_detector.compute_gaze(frame)
        position_results = position_analyzer.analyze_positions(frame)
        lip_results = lip_analyzer.analyze_frame(frame)
        distance_results = distance_estimator.compute_distance(frame)

        # Visualization with adjusted offsets
        visualize_gaze(frame, gaze_results, offset_y=30)  # Add gaze visualization
        visualize_head(frame, position_results, offset_y=90)  # Add head position visualization
        visualize_lip_movement(frame, lip_results, offset_y=150)  # Add lip movement visualization
        visualize_distance(frame, distance_results)  # Distance annotations remain dynamic

        # Display the frame
        cv2.imshow("Multimodal Detection and Visualization", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
