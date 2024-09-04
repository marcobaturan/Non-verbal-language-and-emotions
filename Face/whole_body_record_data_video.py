"""# Biometric, gestural, postural, facial, and ocular information extraction.

# This project aims to extract gestures, postures, expressions, emotions, movements,
# reactions, and behaviors of one or more people in real-time, and store them in a dataset for machine learning.

# Includes:
# - Gestures
# - Facial expressions
# - Postures
# - Blinking
# - Eye tracking
# - Pulse
# - Temperature
# - Emotions
# - Pupil contraction

# Sources:
# - https://github.com/rwightman/posenet-python
# - https://medium.com/@srivastavashivansh8922/building-a-gesture-and-posture-recognition-system-using-python-and-mediapipe-0ef04272a6d2
# - https://www.geeksforgeeks.org/posenet-pose-estimation/
# - https://github.com/AaTekle/Body-Language-Decoder
# - https://github.com/CMU-Perceptual-Computing-Lab/openpose
# - https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
# - https://www.google.com/search?q=estimate+body+temp+from++hear+rate&client=ubuntu&hs=YxD&sca_esv=c7a01837b1eb8e0f&sca_upv=1&channel=fs&sxsrf=ADLYWIInoMM-htcSc3_zZOSFNGsU5uQa4A%3A1725451393687&ei=gUzYZvPTKbu7xc8PseK6uAw&ved=0ahUKEwizmJern6mIAxW7XfEDHTGxDscQ4dUDCBA&uact=5&oq=estimate+body+temp+from++hear+rate&gs_lp=Egxnd3Mtd2l6LXNlcnAiImVzdGltYXRlIGJvZHkgdGVtcCBmcm9tICBoZWFyIHJhdGUyCBAAGIAEGKIEMggQABiABBiiBDIIEAAYgAQYogQyCBAAGIAEGKIESOQJUMoHWMoHcAJ4AJABAJgBfqABfqoBAzAuMbgBA8gBAPgBAZgCA6ACnAHCAgsQABiABBiwAxiiBJgDAOIDBRIBMSBAiAYBkAYEkgcDMi4xoAfiBA&sclient=gws-wiz-serp
# - https://usariem.health.mil/index.cfm/research/products/cbt_algorithm
# - https://journals.humankinetics.com/view/journals/jmpb/1/2/article-p79.xml
# - https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2022.882254/full
"""



from keras.models import load_model  # TensorFlow es necesario para que Keras funcione
import cv2  # Requires installation of opencv-python
import numpy as np
from datetime import datetime
import pandas as pd
import time
import scipy.fftpack
import mediapipe as mp
from math import sqrt

# CSV file to store the collected data
csv_file = 'biometrics.csv'

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the pre-trained model
model = load_model("keras_model.h5", compile=False)

# Load the emotion labels
class_names = open("labels.txt", "r").readlines()

# The camera can be 0 or 1 depending on the default camera of your computer
camera = cv2.VideoCapture(0)

# Parameters for pulse detection
heartbeat_count = 128
heartbeat_values = [0] * heartbeat_count
heartbeat_times = [time.time()] * heartbeat_count
x, y, w, h = 950, 300, 100, 100

# Constants for blink detection
COUNTER = 0
TOTAL_BLINKS = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Mediapipe Face Mesh for blink detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.7)

# Initialize a DataFrame to store the results
df = pd.DataFrame(columns=['timestamp', 'emotion', 'pulse', 'Temperature', 'blinks', 'gesture','posture','head movement','faces','pupils'])

def emotion_detection(image):
    """
    Predict the emotion based on the input image.
    """
    # Predict the emotion
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name.strip(), confidence_score

def extract_pulse(cap):
    global heartbeat_values, heartbeat_times
    """
    Extract the pulse rate from the video capture using image processing and FFT.
    """
    # Capture the frame and process the image
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop_img = img[y:y + h, x:x + w]
    
    # Update the pulse data
    heartbeat_values = heartbeat_values[1:] + [np.mean(crop_img)]
    heartbeat_times = heartbeat_times[1:] + [time.time()]
    
    # Apply a simple low-pass filter
    filtered_values = np.convolve(heartbeat_values, np.ones(3)/3, mode='same')
    
    # Calculate the FFT to find the dominant frequency
    fft_values = np.abs(scipy.fftpack.fft(filtered_values - np.mean(filtered_values)))
    freqs = scipy.fftpack.fftfreq(len(fft_values), d=(heartbeat_times[-1] - heartbeat_times[0]) / len(fft_values))
    
    # Find the dominant frequency within the typical pulse range (0.8 - 3 Hz, equivalent to 48 - 180 BPM)
    freq_in_bpm = np.abs(freqs * 60)
    valid_idxs = np.where((freq_in_bpm > 48) & (freq_in_bpm < 180))
    dominant_freq_idx = valid_idxs[0][np.argmax(fft_values[valid_idxs])]
    bpm = freq_in_bpm[dominant_freq_idx]
    
    return int(bpm)

def estimate_core_temp(hr_value):
    """
    Estimate the core body temperature based on the heart rate.
    """
    start_temp = 37.0
    estimated_temp = start_temp + (hr_value - 70) * 0.01  # Simplified calculation
    return round(estimated_temp, 2)

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def blink_ratio(landmarks, right_indices, left_indices):
    """
    Calculate the blink ratio to detect blinks.
    """
    # Calcular distancias para el ojo derecho
    right_eye_horiz_dist = euclidean_distance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    right_eye_vert_dist = euclidean_distance(landmarks[right_indices[12]], landmarks[right_indices[4]])
    # Calcular distancias para el ojo izquierdo
    left_eye_horiz_dist = euclidean_distance(landmarks[left_indices[0]], landmarks[left_indices[8]])
    left_eye_vert_dist = euclidean_distance(landmarks[left_indices[12]], landmarks[left_indices[4]])
    # Calcular la relación
    right_eye_ratio = right_eye_horiz_dist / right_eye_vert_dist
    left_eye_ratio = left_eye_horiz_dist / left_eye_vert_dist
    return (right_eye_ratio + left_eye_ratio) / 2

def landmarks_detection(image, results):
    """
    Detect facial landmarks and return them as coordinates.
    """
    img_height, img_width = image.shape[:2]
    return [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection

# Define gesture recognition function
def recognize_gesture(hand_landmarks):
    """
    Recognize hand gestures based on the landmarks detected by Mediapipe.
    """
    thumb_tip = hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks[mp_holistic.HandLandmark.PINKY_TIP].y
    wrist = hand_landmarks[mp_holistic.HandLandmark.WRIST].y

    # Count how many fingers are raised
    fingers_up = sum([
        thumb_tip < wrist,  # Thumb raised
        index_tip < wrist,  # Index finger raised
        middle_tip < wrist, # Middle finger raised
        ring_tip < wrist,   # Ring finger raised
        pinky_tip < wrist   # Pinky finger raised
    ])

    # Recognize specific gestures
    if fingers_up == 5:
        return "Open Hand"
    elif fingers_up == 1 and index_tip < wrist:
        return "Pointing"
    elif fingers_up == 2 and thumb_tip > index_tip and middle_tip > index_tip:
        return "Peace Sign"
    elif fingers_up == 0:
        return "Closed Fist"
    elif thumb_tip < wrist and index_tip < wrist:
        return "Thumbs Up"
    elif thumb_tip < wrist and pinky_tip < wrist:
        return "Shaka Sign"
    else:
        return fingers_up

# Define posture recognition function
def recognize_posture(pose_landmarks):
    """
    Recognize the posture of the person based on the pose landmarks.
    """
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    # Calculate the angle between shoulders and hips
    if left_hip.y > 0.7 and right_hip.y > 0.7:
        return "Sit"
    elif left_shoulder.y < left_hip.y and right_shoulder.y < right_hip.y:
        return "Stand"
    else:
        return "Unknown posture"

    # detect shrug shoulders
    if abs(left_shoulder.y - right_shoulder.y) < 0.05:
        return "shrug shoulders"
    elif left_shoulder.y < left_hip.y and right_shoulder.y < right_hip.y:
        return "up arms"
    else:
        return "Neutral posture"

# Define head movement recognition function
def recognize_head_movement(face_landmarks):
    nose_tip = face_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
    left_ear = face_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
    right_ear = face_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]

    # Asentir (mover la cabeza hacia arriba y abajo)
    if abs(nose_tip.y - left_ear.y) > 0.1 and abs(nose_tip.y - right_ear.y) > 0.1:
        return "affirmative"
    # Negar (mover la cabeza de lado a lado)
    elif abs(left_ear.x - right_ear.x) > 0.1:
        return "Negative"
    else:
        return "Neutral head"
    
# Init classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame

def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints

def determine_direction(keypoints, eye_width, eye_height):
    if not keypoints:
        return "Eyes Closed"
    
    x, y = keypoints[0].pt
    if x < eye_width / 3 and y < eye_height / 3:
        return "Top Left"
    elif x > 2 * eye_width / 3 and y < eye_height / 3:
        return "Top Right"
    elif x < eye_width / 3 and y > 2 * eye_height / 3:
        return "Bottom Left"
    elif x > 2 * eye_width / 3 and y > 2 * eye_height / 3:
        return "Bottom Right"
    elif x < eye_width / 3:
        return "Left"
    elif x > 2 * eye_width / 3:
        return "Right"
    elif y < eye_height / 3:
        return "Up"
    elif y > 2 * eye_height / 3:
        return "Down"
    else:
        return "Center"

def detect_pupil_direction(frame):
    face_frame = detect_faces(frame, face_cascade)
    if face_frame is not None:
        eyes = detect_eyes(face_frame, eye_cascade)
        directions = []
        for eye in eyes:
            if eye is not None:
                eye = cut_eyebrows(eye)
                threshold = 30  # Threshold value can be adjusted
                keypoints = blob_process(eye, threshold, detector)
                if keypoints:
                    direction = determine_direction(keypoints, eye.shape[1], eye.shape[0])
                    directions.append(direction)
        return directions if directions else ["No Eyes Detected"]
    return ["No Face Detected"]


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        # detect frame from camera
        ret, image = camera.read()
        #success, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        pupils = detect_pupil_direction(image)
        #print("Pupil Directions:", directions)
        

        # Process image and find landmarks
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        holistic_results = holistic.process(image_rgb)
        face_results = face_detection.process(image_rgb)

        # Draw pose and hand landmarks on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if holistic_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
        if holistic_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

        # Recognize gesture
        if holistic_results.right_hand_landmarks:
            gesture = recognize_gesture(holistic_results.right_hand_landmarks.landmark)
            cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            gesture='None'

        # Recognize posture
        if pose_results.pose_landmarks:
            posture = recognize_posture(pose_results.pose_landmarks)
            cv2.putText(image, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Recognize head movement
        if holistic_results.face_landmarks:
            head_movement = recognize_head_movement(holistic_results.face_landmarks)
            cv2.putText(image, head_movement, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Count faces detected
        if face_results.detections:
            num_faces = len(face_results.detections)
            cv2.putText(image, f"{num_faces} rostro(s) detectado(s)", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image
        cv2.imshow('Gesture, Posture, and Face Detection', image)
        
        # redim picture
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # show emotion in window
        cv2.imshow("Emotion face", image_resized)
        
        # Pic to array
        image_np = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        
        # normalization array
        image_np = (image_np / 127.5) - 1
        
        # detect and record emotion
        emotion, confidence = emotion_detection(image_np)
        
        # detect pulse
        pulse = extract_pulse(camera)
        
        # detect temp
        temp = estimate_core_temp(pulse)
        
        # detect blink
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = landmarks_detection(image, results)
            eyes_ratio = blink_ratio(landmarks, RIGHT_EYE, LEFT_EYE)
            
            if eyes_ratio > 3:
                COUNTER += 1
            else:
                if COUNTER > 4:
                    TOTAL_BLINKS += 1
                    COUNTER = 0
            
            cv2.putText(image, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONT, 1, (0, 255, 0), 2)

        # save metrics
        now = datetime.now()
        df = df._append({
            'timestamp': now.strftime("%d/%m/%Y %H:%M:%S"),
            'emotion': emotion,
            'pulse': pulse,
            'Temperature': temp,
            'blinks': TOTAL_BLINKS,
            'gesture':gesture,
            'posture':posture,
            'head movement': head_movement,
            'faces':num_faces,
            'pupils':pupils,
        }, ignore_index=True)
        
        # detect keyboard
        keyboard_input = cv2.waitKey(1)
        
        # pulse ESC
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

    # save data
    df.to_csv(csv_file, index=False)

