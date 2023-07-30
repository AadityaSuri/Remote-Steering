from Model.Gesture_classification import GestureFNN
import mediapipe as mp
import torch
import cv2 as cv


def flatten_scale_relative(results):
    flattened_scaled_results = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_wrist = hand_landmarks.landmark[0].x
        y_wrist = hand_landmarks.landmark[0].y

        for landmark in hand_landmarks.landmark:
            flattened_scaled_results.append(landmark.x - x_wrist)
            flattened_scaled_results.append(landmark.y - y_wrist)
            flattened_scaled_results.append(landmark.z)

    return flattened_scaled_results

def predict_gesture(model, results):
    tensor_results = torch.tensor(flatten_scale_relative(results))
    tensor_results = tensor_results.unsqueeze(0)
    # tensor_results = tensor_results.unsqueeze(0)
    tensor_results = tensor_results.float()

    output = model(tensor_results)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

def main():
    print("Starting...")
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                                     min_tracking_confidence=0.5)

    cap = cv.VideoCapture(0)
    model = GestureFNN(input_dim=126, hidden_dim_1=100, hidden_dim_2=64, output_dim=3)
    model.load_state_dict(torch.load('Training/Model/model.pth'))
    model.eval()

    print("Camera started...")
    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        results = hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                predicted = predict_gesture(model, results)
                print(predicted)
                cv.putText(img, str(predicted), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow("Image", img)

        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
