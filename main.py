import cv2
import mediapipe as mp
import cv2 as cv
from enum import Enum

class HandLandmarks(Enum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


def main():

    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv.VideoCapture(0)

    # img = cv.imread('istockphoto-1267273871-612x612.jpg')


    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)

        # resize the image
        img = cv.resize(img, (700, 500), fx=0.5, fy=0.5)

        if not success:
            break

        results = hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            img = draw_landmarks(results, img)


        cv.imshow("image", img)
        cv.waitKey(1)

    cap.release()

    # results = hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # hand_landmarks = results.multi_hand_landmarks
    #
    # print(hand_landmarks[0])
    # print("!!!!!!!!!!!!!!!!!!!!!!")
    # print(hand_landmarks[1])
    # print(len(hand_landmarks))
    # print(type(hand_landmarks))
    # print(hand_landmarks[0].landmark[0])
    #
    # img = draw_landmarks(results, img)
    #
    #
    # cv.imshow('image', img)
    # cv.waitKey(0)
    cv.destroyAllWindows()

def draw_landmarks(detection_results, image):

    width = image.shape[1]
    height = image.shape[0]

    for hand_landmarks in detection_results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x = min(int(landmark.x * width), width - 1)
            y = min(int(landmark.y * height), height - 1)
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)

    return image



if __name__ == '__main__':
    main()