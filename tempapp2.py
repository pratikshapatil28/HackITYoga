import numpy as np
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from keras.models import load_model
import update

from camera_module import cap
# from app import cap
app = Flask(__name__)


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False

model = load_model("C:\\Users\\hp\\OneDrive\\Desktop\\myenv\\model1.h5")
label = np.load("C:\\Users\\hp\\OneDrive\\Desktop\\myenv\\labels1.npy")



def generate_frames1():
    while True:
        lst = []

        _, frm = cap.read()

        frm = cv2.flip(frm, 1)

        res = holistic.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        frm = cv2.blur(frm, (4, 4))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst_array = np.array(lst).reshape(1, -1)
            lst_reshaped = lst_array.reshape((lst_array.shape[0], 1, lst_array.shape[1]))

            # Predict using the reshaped input
            p = model.predict(lst_reshaped)
            pred = label[np.argmax(p)]
            
            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
                if pred=="tree":
                    if res.pose_landmarks:
                        landmarks = res.pose_landmarks.landmark
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

                        # fshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        #             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                        #             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                        # felbow = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        #         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        #         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                        # fwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        #         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                        #         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                        fshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                        felbow = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
                        fwrist = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                        fangle = update.calculate_angle(fshoulder, felbow, fwrist)

                        
                        acc0 = update.accuracy(angl,45)
                        acc1 = update.accuracy(angler,45)
                        acc2 = update.accuracy(fangle,25)
                        # if not (<angl< 185 or 175 < angler < 185):
                        #     cv2.putText(frm, "Correct hand posture", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        if not (20<fangle<30):
                            cv2.putText(frm, "Correct your right leg", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        acc = (acc0+ acc1 + acc2) / 3
                        # print(acc2)
                        cv2.putText(frm, f'Accuracy: {acc2}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Rest of your code for drawing landmarks and calculating angles
                if pred=="warrior":
                    if res.pose_landmarks:
                        landmarks = res.pose_landmarks.landmark
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
                        cv2.putText(frm, str(fangle), 
                                tuple(np.multiply(felbow[:2], [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.putText(frm, str(angl), 
                                tuple(np.multiply(felbow[:2], [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.putText(frm, str(angler), 
                                tuple(np.multiply(felbow[:2], [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.putText(frm, str(srangle), 
                                tuple(np.multiply(felbow[:2], [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
                        cv2.putText(frm, str(slangle), 
                                tuple(np.multiply(felbow[:2], [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
                        # print("ang",angl)
                        # print(angler)
                        # print(fangle)
                        # print(langle)
                        # print(srangle)
                        print(slangle)
                        acc0 = update.accuracy(angl,150)
                        acc1 = update.accuracy(angler,150)
                        acc2 = update.accuracy(fangle,110)
                        acc3 = update.accuracy(langle,90)
                        acc4 = update.accuracy(srangle,90)
                        acc5 = update.accuracy(slangle,90)
                        # print("acc0 = ",acc0)
                        # print("acc1 = ",acc1)
                        # print("acc2 = ",acc2)
                        # print("acc3 = ",acc3)
                        # print("acc4 = ",acc4)
                        # print("acc5 = ",acc5)
                        
                        acc = (acc0+ acc1 + acc2 + acc3 + acc4 + acc5) / 6
                        # print("acc = ",acc)
                        # print(acc)
                        if not (175<angl< 185 or 175 < angler < 185):
                            cv2.putText(frm, "Correct hand posture", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        if not (20<fangle<30):
                            cv2.putText(frm, "Correct your right leg", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        if not (80<srangle<110 or 80 < slangle < 110):
                            cv2.putText(frm, "Correct Shoulder posture", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        if not (85<langle<95):
                            cv2.putText(frm, "Correct your left leg", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
 
                        # if (40 < angl < 70) and (40 < angler < 70) and (25 < fangle < 80):
                        # if (30 < angl < 50) and (30 < angler < 50) and (40 < fangle < 50):   
                        #     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        #                                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        #                                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                        #                                 )
                        cv2.putText(frm, f'Accuracy: {int(acc)}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            
            else:
                cv2.putText(frm, "Asana is either wrong or not trained", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)
        else:
            cv2.putText(frm, "Make Sure Full body is visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # Drawing landmarks
        mp_drawing.draw_landmarks(frm, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

        ret, encoded_frame = cv2.imencode('.jpg', frm)
        if not ret:
            print("Error: Failed to encode frame")
            continue

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+ encoded_frame.tobytes() + b'\r\n')

@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
