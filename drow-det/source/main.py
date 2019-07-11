from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
# import argparse
import imutils
import time
import dlib
import cv2


def s_alarm(path):
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    # Compute the euclidean distances between two sets of
    # vertical eye landmark points
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])


    # Compute the euclidean distance between
    # horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    eye_aspr = (A + B) / (2.0 * C)

    return eye_aspr

# Main Function below
print("Welcome to drowsiness prediction")
shape_pred_path = "shape_pr.dat"              # str(input("Enter the path to Facial Landmark Predictor: "))
audio_alarm_path = "sound.wav"
index_webcam = int(input("Please enter the index of the webcam / video source on the system(integer): "))

# Some global constant variables

# EYE_AR_THRESH: It is the threshold eye aspect ratio below which a blink is counted
# MAX_EYE_OFF_FRAMES: Maximum number of consecutive frames for which an eye can closed, after this alarm is triggered
EYE_AR_THRESH = 0.3
MAX_EYE_OFF_FRAMES = 48

# COUNTER: Frame counter to count number of frames
# ALARM_ON: State of alarm
COUNTER = 0
ALARM_ON = False
# If COUNTER exceeds MAX_EYE_OFF_FRAMES, ALARM_ON will be turned to True

# Initialize dlib's HOG face detector
print("[INFO: These are only in developement version, and will be removed in the Production version] Loading Facial Landmark Predictor")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_pred_path)

# Extract the indexes of left eye and right eye using array slicing
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Start the video stream
print("[INFO: These are only in developement version, and will be removed in the Production version] Starting Video Stream Thread")

# Start the video stream on the provided webcam index
vs = VideoStream(src=index_webcam).start()
# Pause the program for a second to allow the camera to warm up
time.sleep(1.0)

# Loop over all the frames in the video stream
while True:
    # Grab the frame from the threaded video file stream
    # Now resize it to a width of 450px, and convert it to grayscale (using cv2)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (in form of rectangles) in the grayscale frame using dlib's frontal face detector
    rects = detector(gray, 0)

    # Now apply facial landmark detection to get the important regions of face

    # For this, loop over all the face detections
    for rect in rects:
        # Determine facial landmarks for face and then convert to numpy (x, y) coordinates array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates then use to compute eye aspect ratio for both eyes

        left_e = shape[l_start:l_end]
        right_e = shape[r_start:r_end]

        # Calculate both eye's aspect ration using the function we defined at the start
        left_eye_aspr = eye_aspect_ratio(left_e)
        right_eye_aspr = eye_aspect_ratio(right_e)

        # Get the average eye aspect ratio
        ear = (left_eye_aspr + right_eye_aspr) / 2.0

        #######################################################################################################
        # Visualization section: Only for Development and Debugging purposes; Remove in the Production version
        #####################################################################################################

        # Compute the Convex hull for both left and right eyes then visualize them
        left_e_hull = cv2.convexHull(left_e)
        right_e_hull = cv2.convexHull(right_e)

        cv2.drawContours(frame, [left_e_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_e_hull], -1, (0, 255, 0), 1)


        # Check to see if eye aspect ratio is below blink threshold, is so, increment blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if eyes were closed for a sufficient number of COUNTERs then sound alarm
            if COUNTER >= MAX_EYE_OFF_FRAMES:
                # If alarm is not already on, then turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # Check to see if alarm file is provided, and then start a thread to have alarm sound played in the background
                    if audio_alarm_path != "":
                        t = Thread(target=s_alarm, args=(audio_alarm_path, ))
                        t.daemon = True
                        t.start()

                # Draw an alarm on the frame
                cv2.putText(frame, "Drowsiness Alert!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, break the loop and exit
        if key == ord('q'):
            break

    # Clean mess up

cv2.destroyAllWindows()
vs.stop()
