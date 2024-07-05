# https://github.com/stebusse/mediapipe-plot-pose-live/blob/main/plot_pose_live.py
import matplotlib.pyplot as plt

# connections for the MediaPipe topology
LANDMARK_GROUPS = [
    [8, 6, 5, 4, 0, 1, 2, 3, 7],   # eyes
    [10, 9],                       # mouth
    [11, 13, 15, 17, 19, 15, 21],  # right arm
    [11, 23, 25, 27, 29, 31, 27],  # right body side
    [12, 14, 16, 18, 20, 16, 22],  # left arm
    [12, 24, 26, 28, 30, 32, 28],  # left body side
    [11, 12],                      # shoulder
    [23, 24],                      # waist
]


def plot_world_landmarks(ax, landmarks, landmark_groups=LANDMARK_GROUPS):
    """_summary_
    Args:
        ax: plot axes
        landmarks  mediapipe
    """

    # skip when no landmarks are detected
    if landmarks is None:
        return

    ax.cla()

    # had to flip the z axis
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(1, -1)

    # get coordinates for each group and plot
    for group in landmark_groups:
        plotX, plotY, plotZ = [], [], []

        plotX = [landmarks.landmark[i].x for i in group]
        plotY = [landmarks.landmark[i].y for i in group]
        plotZ = [landmarks.landmark[i].z for i in group]

        # this can be changed according to your camera
        ax.plot(plotX, plotZ, plotY)

    plt.pause(.001)
    return

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# create axes like this
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from MacBook camera
cap = cv2.VideoCapture(1)  # Use index 1 for MacBook camera

if not cap.isOpened():
    print("Error: Camera not accessible or not found!")
    exit()

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")

# Video writer to save the output
out = cv2.VideoWriter('output_with_landmarks.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# List to store landmarks and timestamps
landmarks_list = []

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Perform pose detection
    results = pose.process(image)

    # Convert frame back to BGR for display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]

        # Scale the coordinates to the resolution of the video frame
        left_ankle_coords = np.multiply(left_ankle, [frame_width, frame_height, 1]).astype(float)
        right_ankle_coords = np.multiply(right_ankle, [frame_width, frame_height, 1]).astype(float)
        left_knee_coords = np.multiply(left_knee, [frame_width, frame_height, 1]).astype(float)
        right_knee_coords = np.multiply(right_knee, [frame_width, frame_height, 1]).astype(float)
        left_hip_coords = np.multiply(left_hip, [frame_width, frame_height, 1]).astype(float)
        right_hip_coords = np.multiply(right_hip, [frame_width, frame_height, 1]).astype(float)

        plot_world_landmarks(ax, results.pose_world_landmarks)

        # Save landmarks with timestamp
        current_time = time.time()
        timestamp = current_time - start_time
        landmarks_list.append({
            'timestamp': timestamp,
            'left_ankle_x': left_ankle_coords[0], 'left_ankle_y': left_ankle_coords[1], 'left_ankle_z': left_ankle_coords[2],
            'right_ankle_x': right_ankle_coords[0], 'right_ankle_y': right_ankle_coords[1], 'right_ankle_z': right_ankle_coords[2],
            'left_knee_x': left_knee_coords[0], 'left_knee_y': left_knee_coords[1], 'left_knee_z': left_knee_coords[2],
            'right_knee_x': right_knee_coords[0], 'right_knee_y': right_knee_coords[1], 'right_knee_z': right_knee_coords[2],
            'left_hip_x': left_hip_coords[0], 'left_hip_y': left_hip_coords[1], 'left_hip_z': left_hip_coords[2],
            'right_hip_x': right_hip_coords[0], 'right_hip_y': right_hip_coords[1], 'right_hip_z': right_hip_coords[2]
        })

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Save frame to video
    out.write(image)

    # Display the resulting frame
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Convert landmarks list to DataFrame and save to CSV
landmarks_df = pd.DataFrame(landmarks_list)
landmarks_df.to_csv('landmarks.csv', index=False)
