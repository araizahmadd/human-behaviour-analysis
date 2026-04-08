import cv2
import mediapipe as mp


class DistanceEstimator:
    def __init__(self, focal_length=640, real_face_height=20):
        """
        Initializes the DistanceEstimator.
        :param focal_length: Camera focal length (default: 640).
        :param real_face_height: Average height of a human face in cm (default: 20).
        """
        self.focal_length = focal_length
        self.real_face_height = real_face_height

    def visualize(frame, results):
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

    @staticmethod
    def get_distance_category(distance):
        """
        Categorizes the distance into predefined ranges.
        :param distance: Calculated distance.
        :return: String category of distance.
        """
        if distance < 45:
            return "Too Close"
        elif 45 <= distance < 90:
            return "Close"
        elif 90 <= distance < 180:
            return "Far"
        else:
            return "Too Far"

    
    def compute_distance(self, frame, face_landmarks_list):
        """
        Computes the distance to a face in the frame.
        :param frame: Input frame (BGR image).
        :param face_landmarks_list: List of face landmarks.
        :return: List of tuples containing distance, distance category, and key coordinates for each detected face.
        """
        output_data = []

        if face_landmarks_list:
            for face_landmarks in face_landmarks_list:
                forehead = face_landmarks[10]
                chin = face_landmarks[152]

                image_height, image_width, _ = frame.shape
                forehead_coords = (int(forehead.x * image_width), int(forehead.y * image_height))
                chin_coords = (int(chin.x * image_width), int(chin.y * image_height))

                face_height_pixels = abs(forehead_coords[1] - chin_coords[1])

                if face_height_pixels > 0:  # Avoid division by zero
                    distance = (self.focal_length * self.real_face_height) / face_height_pixels
                    distance_category = self.get_distance_category(distance)
                    output_data.append((distance, distance_category, forehead_coords, chin_coords))

        return output_data


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    estimator = DistanceEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = estimator.compute_distance(frame)

        for distance, category, forehead_coords, chin_coords in results:
            cv2.circle(frame, forehead_coords, 5, (0, 255, 0), -1)
            cv2.circle(frame, chin_coords, 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Distance: {int(distance)}',
                        (forehead_coords[0] + 10, forehead_coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f'Category: {category}',
                        (forehead_coords[0] + 10, forehead_coords[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow('FaceMesh - Distance Estimation', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()