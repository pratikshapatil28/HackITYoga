import numpy as np
import math


# cap = cv2.VideoCapture(0)

def accuracy(angle,a):
    # Step 1: Calculate the absolute difference
    diff = abs(angle - a)
    #print(diff)
    # Step 2: Convert difference to error percentage
    err = (diff / a) * 100
    #print(err)
    # Step 3: Calculate accuracy
    acc = 100 - err
   
    # Step 4: Ensure accuracy is within the range [0, 100]
    acc = max(0, min(int(acc), 100))
    #print(acc)
    return acc

def calculate_angle(a,b,c):
            a = np.array(a) # First
            b = np.array(b) # Mid
            c = np.array(c) # Endq
            #print(a)
            vector1 = [b[0] - c[0], b[1] - c[1],b[2]-c[2]]
            vector2 = [b[0] - a[0], b[1] - a[1],b[2]-a[2]]
            #print(vector1)
            dot_product = sum(x * y for x, y in zip(vector1, vector2))

    # Calculate magnitudes
            magnitude1 = math.sqrt(sum(x**2 for x in vector1))
            magnitude2 = math.sqrt(sum(y**2 for y in vector2))

            # Calculate angle in radians
            angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))
            angle = math.degrees(angle_radians)
            
            if angle >180.0:
                angle = 360-angle
            #print(angle)
            return angle
        
# def process_pose_landmarks(landmarks,pose_type):
#     if pose_type == 'left':
#         shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
#                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
#         elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
#                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
#         wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
#                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
#         angl = calculate_angle(shoulder, elbow, wrist)
#         acc0 = accuracy(angl, 45)
#         return acc0
#         # Additional logic for feedback based on 'left' pose

#     elif pose_type == 'right':
#         # Similar logic for processing right pose
#         rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
#         relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
#         rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
#         angler = calculate_angle(rshoulder, relbow, rwrist)
#         acc1 = accuracy(angler, 45)
#         return acc1
#     elif pose_type == 'front':
#         # Similar logic for processing front pose
#         fshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
#         felbow = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
#         fwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
#         fangle = calculate_angle(fshoulder, felbow, fwrist)
#         acc2 = accuracy(fangle, 45)
#         return acc2
#     else:
#         # Handle unsupported pose types
#         pass
# import threading
