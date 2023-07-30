from Model.Gesture_classification import GestureFNN
import mediapipe as mp
import cv2 as cv
from utils.GestureDetection import GestureClassifier


def main():
    print("Starting...")
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                                     min_tracking_confidence=0.5)
    gesture_classifier = GestureClassifier(GestureFNN(input_dim=126, hidden_dim_1=100, hidden_dim_2=64, output_dim=3),
                                           'Training/Gesture_detection/Model/model.pth'
                                           ,hands)

    cap = cv.VideoCapture(0)
    print("Camera started...")
    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        img = gesture_classifier.draw_prediction(img)
        cv.imshow("Image", img)

        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
