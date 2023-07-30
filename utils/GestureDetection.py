import numpy as np
import torch
import cv2 as cv


class GestureClassifier:
    def __init__(self, model: torch.nn.Module, weights_path: str, mediapipe_hands: object) -> None:
        self.model = model
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.mediapipe_hands = mediapipe_hands


    def flatten_scale_relative(self, results: object) -> np.ndarray:
        return np.array([(landmark.x - hand_landmarks.landmark[0].x,
                          landmark.y - hand_landmarks.landmark[0].y,
                          landmark.z) for hand_landmarks in results.multi_hand_landmarks
                         for landmark in hand_landmarks.landmark]).flatten()

    def find_center_of_hand(self, hand_landmarks: object) -> np.ndarray:
        landmarks_array = np.array([(landmark.x, landmark.y, landmark.z)
                                    for landmark in hand_landmarks.landmark])
        return landmarks_array.mean(axis=0)

    def calculate_angle(self, L_coordinates: tuple, R_coordinates: tuple) -> float:
        x1, y1, z1 = L_coordinates
        x2, y2, z2 = R_coordinates

        y_diff = y2 - y1
        x_diff = x2 - x1

        if x_diff == 0:
            if y_diff >= 0:
                return 90
            else:
                return -90

        theta_rad = np.arctan2(y_diff, x_diff)
        theta_degrees = np.degrees(theta_rad)

        if theta_degrees > 90:
            theta_degrees = 180 - theta_degrees
        elif theta_degrees < -90:
            theta_degrees = -(180 + theta_degrees)

        return round(theta_degrees, 2)

    def predict_gesture(self, results: object) -> int:
        tensor_results = torch.tensor(self.flatten_scale_relative(results)).unsqueeze(0).float()

        output = self.model(tensor_results)
        softmax_output = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.item()

        if predicted == 0 or predicted == 1:
            if softmax_output[0][predicted] < 0.9:
                predicted = 2

        return predicted

    def draw_prediction(self, img: np.ndarray) -> np.ndarray:
        results = self.mediapipe_hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                predicted = self.predict_gesture(results)
                if predicted == 0:
                    x_center_L, y_center_L, z_center_L = self.find_center_of_hand(results.multi_hand_landmarks[0])
                    x_center_R, y_center_R, z_center_R = self.find_center_of_hand(results.multi_hand_landmarks[1])

                    cv.line(img, (int(x_center_L * img.shape[1]), int(y_center_L * img.shape[0])),
                            (int(x_center_R * img.shape[1]), int(y_center_R * img.shape[0])), (0, 0, 255), 2)

                    angle = self.calculate_angle((x_center_L, y_center_L, z_center_L), (x_center_R, y_center_R, z_center_R))
                    cv.putText(img, str(angle), (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                    cv.putText(img, "DRIVING", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                elif predicted == 1:
                    cv.putText(img, "STOP", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                elif predicted == 2:
                    cv.putText(img, "UNKNOWN", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        return img
