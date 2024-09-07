import os
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pandas as pd
from io import BytesIO

# Define the path where files will be saved
save_path = "C:/Users/patha/Desktop/Experiments/Hackathon"

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

# Load MediaPipe pose estimator
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize session state to store recorded data
if "recorded_release_angles" not in st.session_state:
    st.session_state.recorded_release_angles = []
if "recorded_shoulder_alignments" not in st.session_state:
    st.session_state.recorded_shoulder_alignments = []

# Define ideal angles for swing ball
ideal_release_angle_range = (15.0, 20.0)  # degrees
ideal_shoulder_alignment_range = (30.0, 45.0)  # degrees

# Navigation bar with pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Bowling Action Analyzer", "About"])

# Home Page
if page == "Home":
    st.title("Welcome to the Bowling Action Analyzer")
    st.write("""
    This application analyzes your bowling action in real-time using your webcam.
    You can get metrics like release angle and shoulder alignment to improve your bowling technique.
    Use the navigation menu to explore the app.
    """)

    # Add option to view recorded data
    if st.button("View Recorded Data"):
        if st.session_state.recorded_release_angles:
            recorded_data = pd.DataFrame({
                "Release Angle": st.session_state.recorded_release_angles,
                "Shoulder Alignment": st.session_state.recorded_shoulder_alignments
            })
            st.write(recorded_data)
        else:
            st.write("No recorded data available.")

# About Page
elif page == "About":
    st.title("About Bowling Action Analyzer")
    st.write("""
    The Bowling Action Analyzer uses computer vision and machine learning techniques to assess 
    bowling biomechanics. Using MediaPipe's pose estimation model, it detects key body landmarks 
    and calculates critical metrics like release angle and shoulder alignment.
    
    This project is designed to help bowlers, coaches, and enthusiasts analyze bowling techniques in real-time.
    """)

# Bowling Action Analyzer Page
elif page == "Bowling Action Analyzer":
    st.title("Bowling Action Analyzer")

    # Create placeholders for the video feed and metrics
    frame_placeholder = st.empty()
    release_angle_display = st.metric("Release Angle", "0.00°")
    shoulder_alignment_display = st.metric("Shoulder Alignment", "0.00°")
    
    # Display the ideal angles for swing ball
    st.sidebar.markdown("### Ideal Angles for Swing Ball")
    st.sidebar.metric("Ideal Release Angle Range", f"{ideal_release_angle_range[0]}° - {ideal_release_angle_range[1]}°")
    st.sidebar.metric("Ideal Shoulder Alignment Range", f"{ideal_shoulder_alignment_range[0]}° - {ideal_shoulder_alignment_range[1]}°")

    # Define function to calculate angles using keypoints
    def calculate_angles(landmarks, side="Right"):
        if side == "Right":
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        else:
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        shoulder_coords = np.array([shoulder.x, shoulder.y])
        elbow_coords = np.array([elbow.x, elbow.y])
        wrist_coords = np.array([wrist.x, wrist.y])

        upper_arm = elbow_coords - shoulder_coords
        forearm = wrist_coords - elbow_coords

        angle_radians = np.arccos(np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm)))
        arm_angle = np.degrees(angle_radians)

        if side == "Right":
            shoulder_alignment = np.degrees(np.arctan2(right_hip.y - shoulder.y, right_hip.x - shoulder.x))
        else:
            shoulder_alignment = np.degrees(np.arctan2(left_hip.y - shoulder.y, left_hip.x - shoulder.x))

        return arm_angle, shoulder_alignment

    # Add Start and Stop buttons
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    # Add toggle to select between Left and Right side
    side = st.radio("Select Side", ("Left", "Right"))

    # Map the selection to the correct side
    side_map = {
        "Left": "Left",
        "Right": "Right"
    }
    selected_side = side_map[side]

    # Start video capture if Start button is pressed
    if start_button:
        # Save previous data to an Excel file and offer it for download
        if st.session_state.recorded_release_angles:
            recorded_data = pd.DataFrame({
                "Release Angle": st.session_state.recorded_release_angles,
                "Shoulder Alignment": st.session_state.recorded_shoulder_alignments
            })

            # Create an in-memory Excel file
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                recorded_data.to_excel(writer, index=False, sheet_name='Bowling Data')

            excel_data = excel_buffer.getvalue()

            # Offer the file for download
            st.download_button(
                label="Download Previous Data",
                data=excel_data,
                file_name="previous_bowling_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Clear previous data
            st.session_state.recorded_release_angles = []
            st.session_state.recorded_shoulder_alignments = []
            st.success("Previous records cleared. Starting new session.")

        cap = cv2.VideoCapture(0)  # Change this if a different camera index is needed
        try:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture image from camera.")
                        break

                    # Convert the image to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process the image and get pose landmarks
                    results = pose.process(image_rgb)

                    # Draw landmarks and connections for the selected side
                    if results.pose_landmarks:
                        if selected_side == "Right":
                            landmarks = [
                                mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                mp_pose.PoseLandmark.RIGHT_ELBOW, 
                                mp_pose.PoseLandmark.RIGHT_WRIST, 
                                mp_pose.PoseLandmark.RIGHT_HIP
                            ]
                            connections = [
                                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP)
                            ]

                        else:
                            landmarks = [
                                mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                mp_pose.PoseLandmark.LEFT_ELBOW, 
                                mp_pose.PoseLandmark.LEFT_WRIST, 
                                mp_pose.PoseLandmark.LEFT_HIP
                            ]
                            connections = [
                                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP)
                            ]

                        # Draw pose connections for the selected side
                        for connection in connections:
                            start = results.pose_landmarks.landmark[connection[0]]
                            end = results.pose_landmarks.landmark[connection[1]]
                            cv2.line(frame, 
                                     (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                                     (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])),
                                     (0, 255, 0), 3)

                        # Draw only the selected landmarks on the frame
                        for landmark_idx in landmarks:
                            landmark = results.pose_landmarks.landmark[landmark_idx]
                            cv2.circle(frame, 
                                       (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 
                                       5, (0, 255, 0), -1)

                        # Calculate the release angle and shoulder alignment for the selected side
                        release_angle, shoulder_alignment = calculate_angles(results.pose_landmarks.landmark, selected_side)

                        # Update Streamlit metrics in real-time
                        release_angle_display.metric("Release Angle", f"{release_angle:.2f}°", delta=f"{release_angle - ideal_release_angle_range[0]:.2f}°")
                        shoulder_alignment_display.metric("Shoulder Alignment", f"{shoulder_alignment:.2f}°", delta=f"{shoulder_alignment - ideal_shoulder_alignment_range[0]:.2f}°")

                        # Record the angles in session state
                        st.session_state.recorded_release_angles.append(release_angle)
                        st.session_state.recorded_shoulder_alignments.append(shoulder_alignment)

                    # Display the video feed in Streamlit
                    frame_placeholder.image(frame, channels="BGR")

                    # Break if the Stop button is pressed
                    if stop_button:
                        break
        finally:
            # Release the video capture and destroy OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

        # Save the data to an Excel file after stopping the camera
        recorded_data = pd.DataFrame({
            "Release Angle": st.session_state.recorded_release_angles,
            "Shoulder Alignment": st.session_state.recorded_shoulder_alignments
        })

        # Create an in-memory Excel file
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            recorded_data.to_excel(writer, index=False, sheet_name='Bowling Data')

        excel_data = excel_buffer.getvalue()

        # Create a download button for the Excel file
        st.download_button(
            label="Download Data",
            data=excel_data,
            file_name="bowling_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
