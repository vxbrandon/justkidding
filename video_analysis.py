


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load the landmarks data
landmarks_df = pd.read_csv('landmarks.csv')

# Initialize the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the limits for the axes based on the data
frame_width = int(max(landmarks_df[['left_ankle_x', 'right_ankle_x', 'left_knee_x', 'right_knee_x', 'left_hip_x', 'right_hip_x']].max()))
frame_height = int(max(landmarks_df[['left_ankle_y', 'right_ankle_y', 'left_knee_y', 'right_knee_y', 'left_hip_y', 'right_hip_y']].max()))

ax.set_xlim(0, frame_width)
ax.set_ylim(0, frame_height)
ax.set_zlim(-1, 1)  # Adjust z-limit based on the expected range of z-values

def update(num, data, scatter):
    ax.clear()
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    ax.set_zlim(-1, 1)  # Adjust z-limit based on the expected range of z-values

    xdata = [
        data['left_ankle_x'][num], data['left_knee_x'][num], data['left_hip_x'][num],
        data['right_hip_x'][num], data['right_knee_x'][num], data['right_ankle_x'][num]
    ]
    ydata = [
        data['left_ankle_y'][num], data['left_knee_y'][num], data['left_hip_y'][num],
        data['right_hip_y'][num], data['right_knee_y'][num], data['right_ankle_y'][num]
    ]
    zdata = [
        data['left_ankle_z'][num], data['left_knee_z'][num], data['left_hip_z'][num],
        data['right_hip_z'][num], data['right_knee_z'][num], data['right_ankle_z'][num]
    ]

    scatter._offsets3d = (xdata, ydata, zdata)
    return scatter,

# Create a scatter object for updating
scatter = ax.scatter([], [], [], c='r', marker='o')

# Adjust the interval and fps to make the video longer
interval = 100  # Interval in milliseconds between frames
fps = 10  # Frames per second

# Calculate total duration
total_frames = len(landmarks_df)
total_duration = (total_frames * interval) / 1000  # Total duration in seconds

print(f'Total frames: {total_frames}, Total duration: {total_duration} seconds')

# Create animation
ani = animation.FuncAnimation(fig, update, frames=total_frames, fargs=(landmarks_df, scatter), interval=interval, blit=False)

# Save the animation as a video file
ani.save('landmarks_3d.mp4', writer='ffmpeg', fps=fps)

plt.show()
