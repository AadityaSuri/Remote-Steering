import numpy as np
import torch
import cv2 as cv


class GestureClassifier:
    def __init__(self, model, weights_path, mediapipe_hands):
        self.model = model
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.mediapipe_hands = mediapipe_hands

    def flatten_scale_relative(self, results):
        flattened_scaled_results = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_wrist = hand_landmarks.landmark[0].x
            y_wrist = hand_landmarks.landmark[0].y

            for landmark in hand_landmarks.landmark:
                flattened_scaled_results.append(landmark.x - x_wrist)
                flattened_scaled_results.append(landmark.y - y_wrist)
                flattened_scaled_results.append(landmark.z)

        return flattened_scaled_results

    def find_center_of_hand(self, hand_landmarks):
        x_vals = [landmark.x for landmark in hand_landmarks.landmark]
        y_vals = [landmark.y for landmark in hand_landmarks.landmark]
        z_vals = [landmark.z for landmark in hand_landmarks.landmark]

        x_center = sum(x_vals) / len(x_vals)
        y_center = sum(y_vals) / len(y_vals)
        z_center = sum(z_vals) / len(z_vals)

        return x_center, y_center, z_center

    def calculate_angle(self, L_coordinates, R_coordinates):
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

    def predict_gesture(self, results):
        tensor_results = torch.tensor(self.flatten_scale_relative(results)).unsqueeze(0).float()

        output = self.model(tensor_results)
        softmax_output = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.item()

        if predicted == 0 or predicted == 1:
            if softmax_output[0][predicted] < 0.9:
                predicted = 2

        return predicted

    def draw_prediction(self, img):
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
