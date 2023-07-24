from Model.Gesture_classification import GestureFNN
import mediapipe as mp
import torch
import cv2 as cv

def scale_relative(results):
    for hand_landmarks in results.multi_hand_landmarks:
        x_wrist = hand_landmarks.landmark[0].x
        y_wrist = hand_landmarks.landmark[0].y
        z_wrist = hand_landmarks.landmark[0].z

        for landmark in hand_landmarks.landmark:
            landmark.x -= x_wrist
            landmark.y -= y_wrist
            landmark.z -= z_wrist

    return results

def flatten_results(results):
    flattened_results = []
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            flattened_results.append(landmark.x)
            flattened_results.append(landmark.y)
            flattened_results.append(landmark.z)

    return flattened_results

def predict_gesture(model, results):
    results = scale_relative(results)
    flattened_results = flatten_results(results)
    tensor_results = torch.tensor(flattened_results)
    tensor_results = tensor_results.unsqueeze(0)
    # tensor_results = tensor_results.unsqueeze(0)
    # tensor_results = tensor_results.float()

    output = model(tensor_results)
    _, predicted = torch.max(output.data, 1)
    return predicted

def main():
    print("Starting...")
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                                     min_tracking_confidence=0.5)

    cap = cv.VideoCapture(0)
    model = GestureFNN(input_dim=126, hidden_dim=84, output_dim=3)
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
