"""
Real-Time Hand Detection and Landmark Tracking

This script captures video from a webcam, detects and tracks human hands using
MediaPipe Hands, draws 21 3D hand landmarks with skeletal connections, computes
padded bounding boxes around detected hands, and overlays a live smoothed FPS
counter and hand count on the mirrored video stream.
"""

import time
import cv2
import mediapipe as mp


def main():
    # Initialize MediaPipe Hands and drawing utilities
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Initialize webcam (index 0 is typically the default built-in camera)
    cap = cv2.VideoCapture(0)

    # Verify webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not access the webcam. Please verify that the camera is connected and permitted.")
        return

    # Tracking variables for smoothed FPS calculation
    prev_time = time.perf_counter()
    fps = 0.0
    smoothing_factor = 0.85  # Weight given to previous smoothed FPS to avoid flicker

    try:
        # Initialize MediaPipe Hands solution
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to receive frame from webcam. Exiting stream.")
                    break

                # Flip horizontally for an intuitive mirrored view
                frame = cv2.flip(frame, 1)

                # MediaPipe requires RGB images, OpenCV captures in BGR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame to detect hands and landmarks
                result = hands.process(rgb_frame)

                # Count detected hands
                num_hands = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0

                # Draw bounding boxes and landmarks for detected hands
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Extract frame dimensions
                        h, w, _ = frame.shape
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0

                        # Calculate bounding box bounds from 21 normalized landmarks
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min = min(x_min, x)
                            y_min = min(y_min, y)
                            x_max = max(x_max, x)
                            y_max = max(y_max, y)

                        # Draw box around hand with 20px padding (clamped to frame boundaries)
                        box_x_min = max(0, x_min - 20)
                        box_y_min = max(0, y_min - 20)
                        box_x_max = min(w - 1, x_max + 20)
                        box_y_max = min(h - 1, y_max + 20)
                        cv2.rectangle(frame, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 0), 2)

                        # Draw the 21 hand landmarks and connecting skeleton
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate smoothed FPS
                current_time = time.perf_counter()
                delta_time = current_time - prev_time
                prev_time = current_time

                if delta_time > 0:
                    instant_fps = 1.0 / delta_time
                    fps = (fps * smoothing_factor) + (instant_fps * (1.0 - smoothing_factor))

                # Overlay live smoothed FPS and detected hands count
                cv2.putText(
                    frame,
                    f"FPS: {int(round(fps))}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"Hands: {num_hands}",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # Show the resulting annotated video stream
                cv2.imshow("Hand Movement Detection", frame)

                # Exit loop when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Release the webcam hardware and destroy any open OpenCV GUI windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
