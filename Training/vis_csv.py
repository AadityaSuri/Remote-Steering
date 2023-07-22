import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the csv file
df = pd.read_csv('idle_coordinates.csv', dtype=np.float32)
print(df)

plt.ion()
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [])

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([-0.5, 0.5])  # MediaPipe's z values are between -0.5 and 0.5

ax.view_init(azim=90, elev=90)  # Set camera to bird's eye view


connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
               (2, 5), (5, 6), (6, 7), (7, 8),  # Index finger
               (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
               (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
               (13, 17), (17, 18), (18, 19), (19, 20), (17, 0)]  # Pinky


iter = df.iterrows()
for (i1, row1), (i2, row2) in zip(iter, iter):
    x_vals_L = [row1[3*i+1] for i in range(21)]
    y_vals_L = [row1[3*i+2] for i in range(21)]
    z_vals_L = [row1[3*i+3] for i in range(21)]

    x_vals_R = [row2[3*i+1] for i in range(21)]
    y_vals_R = [row2[3*i+2] for i in range(21)]
    z_vals_R = [row2[3*i+3] for i in range(21)]

    ax.scatter(x_vals_L, y_vals_L, z_vals_L, c='g', s=35)
    ax.scatter(x_vals_R, y_vals_R, z_vals_R, c='g', s=35)

    for connection in connections:
        ax.plot([x_vals_L[connection[0]], x_vals_L[connection[1]]],
                [y_vals_L[connection[0]], y_vals_L[connection[1]]],
                [z_vals_L[connection[0]], z_vals_L[connection[1]]], 'r')

        ax.plot([x_vals_R[connection[0]], x_vals_R[connection[1]]],
                [y_vals_R[connection[0]], y_vals_R[connection[1]]],
                [z_vals_R[connection[0]], z_vals_R[connection[1]]], 'r')

    plt.draw()
    plt.pause(0.005)

    ax.clear()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([-0.5, 0.5])  # MediaPipe's z values are between -0.5 and 0.5

plt.ioff()




