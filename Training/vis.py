import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
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



def draw_landmarks(detection_results, image):
    width = image.shape[1]
    height = image.shape[0]

    for hand_landmarks in detection_results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x = min(int(landmark.x * width), width - 1)
            y = min(int(landmark.y * height), height - 1)
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)

    return image


connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
               (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
               (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
               (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
               (0, 17), (17, 18), (18, 19), (19, 20)]  # Pinky

def find_center_of_hand(hand_landmarks):
    x_vals = [landmark.x for landmark in hand_landmarks.landmark]
    y_vals = [landmark.y for landmark in hand_landmarks.landmark]
    z_vals = [landmark.z for landmark in hand_landmarks.landmark]

    x_center = sum(x_vals) / len(x_vals)
    y_center = sum(y_vals) / len(y_vals)
    z_center = sum(z_vals) / len(z_vals)

    return x_center, y_center, z_center


    


def main():
    cap = cv.VideoCapture(0)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                                     min_tracking_confidence=0.5)

    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter([], [], [])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([-0.5, 0.5])  # MediaPipe's z values are between -0.5 and 0.5

    ax.view_init(azim=90, elev=90)  # Set camera to bird's eye view

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)

        if not success:
            print("Could not read frame")
            break

        results = hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            ax.clear()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([-0.5, 0.5])  # MediaPipe's z values are between -0.5 and 0.5

            for hand_landmarks in results.multi_hand_landmarks:
                x_vals = [landmark.x for landmark in hand_landmarks.landmark]
                y_vals = [landmark.y for landmark in hand_landmarks.landmark]
                z_vals = [landmark.z for landmark in hand_landmarks.landmark]

                x_center, y_center, z_center = find_center_of_hand(hand_landmarks)
                # print(x_center, y_center, z_center)

                # Plot center of hand
                ax.scatter(x_center, y_center, z_center, c='b', s=50)

                ax.scatter(x_vals, y_vals, z_vals, c='g', s=35)

                # Plot connections
                for connection in connections:
                    ax.plot([x_vals[connection[0]], x_vals[connection[1]]],
                            [y_vals[connection[0]], y_vals[connection[1]]],
                            [z_vals[connection[0]], z_vals[connection[1]]], 'r')

            if len(results.multi_hand_landmarks) == 2:
                xcenter1, ycenter1, zcenter1 = find_center_of_hand(results.multi_hand_landmarks[0])
                xcenter2, ycenter2, zcenter2 = find_center_of_hand(results.multi_hand_landmarks[1])

                ax.plot([xcenter1, xcenter2], [ycenter1, ycenter2], [zcenter1, zcenter2], 'b')


            plt.draw()
            plt.pause(0.01)  # Add a short pause to allow the plot to update

            img = draw_landmarks(results, img)

        cv.imshow("image", img)

        if cv.waitKey(1) == 27:  # Press ESC to exit
            break

    cap.release()
    cv.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode


if __name__ == '__main__':
    main()
