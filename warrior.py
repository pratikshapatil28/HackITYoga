
import numpy as np
from flask import Flask, render_template, Response

import cv2


import mediapipe as mp
from flask_pymongo import PyMongo
import bcrypt
import update
from camera_module import cap
app = Flask(__name__)
app.secret_key = 'mysecret'
app.config['MONGO_DBNAME'] = 'db'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/db'

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

mongo = PyMongo(app)

def generate_frame(user_email):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Convert the frame to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with Mediapipe
            results = holistic.process(rgb_frame)

            # Draw landmarks on the frame
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                angl = update.calculate_angle(shoulder, elbow, wrist)
                
                rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                angler = update.calculate_angle(rshoulder, relbow, rwrist)

                fshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                felbow = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                fwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                fangle = update.calculate_angle(fshoulder, felbow, fwrist)
                lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                langle = update.calculate_angle(lshoulder, lelbow, lwrist)
                srangle=update.calculate_angle(fshoulder,rshoulder,relbow)
                slangle=update.calculate_angle(lshoulder,shoulder,elbow)
                cv2.putText(frame, str(fangle), 
                           tuple(np.multiply(felbow[:2], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )
                acc0 = update.accuracy(angl,180)
                acc1 = update.accuracy(angler,180)
                acc2 = update.accuracy(fangle,90)
                acc3 = update.accuracy(langle,180)
                acc4 = update.accuracy(srangle,90)
                acc5 = update.accuracy(slangle,90)
                acc = (acc0+ acc1 + acc2 + acc3 + acc4 + acc5) / 3
                
                # print(acc)
                if not (175<angl< 185 or 175 < angler < 185):
                    cv2.putText(frame, "Correct hand posture", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                if not (20<fangle<30):
                    cv2.putText(frame, "Correct your right leg", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                if not (80<srangle<110 or 80 < slangle < 110):
                    cv2.putText(frame, "Correct Shoulder posture", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                if not (20<langle<30):
                    cv2.putText(frame, "Correct your left leg", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                if user_email:
                    users = mongo.db.get_collection('tab')
                    login_user = users.find_one({'email': user_email})

                    if login_user:
                        accuracies = login_user.get('accuracies', [])

            # Check if 'accuracies' field exists, create it if not
                        if 'accuracies' not in login_user:
                            users.update_one({'email': user_email}, {"$set": {"accuracies": []}})
                            accuracies = []

            # Append the new accuracy
                        accuracies.append(acc)

            # Update the 'accuracies' field in the user document
                        users.update_one({'email': user_email}, {"$set": {"accuracies": accuracies}})
                        
            #
                
                # if (40 < angl < 70) and (40 < angler < 70) and (25 < fangle < 80):
                # if (30 < angl < 50) and (30 < angler < 50) and (40 < fangle < 50):   
                #     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                #                                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                #                                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                #                                 )
                cv2.putText(frame, f'Accuracy: {acc}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                
            # Encode the frame as JPEG
            ret, encoded_frame = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Failed to encode frame")
                continue

            # Yield the frame for streaming
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
            
            # After generating frames, calculate accuracy
            # accuracy = calculate_accuracy()
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
             # Store accuracy in MongoDB
             
          
    app.secret_key = 'mysecret'
    

