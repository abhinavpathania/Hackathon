import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Initialize Mediapipe pose class and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Second point (elbow)
    c = np.array(c)  # Third point (wrist)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# Streamlit app setup
st.title("Cricket Bowling Action Analyzer")
st.write("This tool analyzes your bowling action in real-time.")

# Start video capture
cap = cv2.VideoCapture(0)

if st.button("Start Camera"):
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert the frame to RGB and process using Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Extract landmarks for shoulder, elbow, wrist
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for right shoulder, elbow, wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate the elbow angle
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            st.write(f"Elbow Angle: {elbow_angle:.2f}")

            # Provide real-time feedback
            if elbow_angle > 15:
                st.warning("Elbow flexion exceeds 15°. Illegal bowling action!")
            else:
                st.success("Bowling action is legal!")

            # Draw the pose annotations on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Show video in Streamlit app
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame)

        # Exit with 'q' key press (for debugging purposes, not used in Streamlit)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
