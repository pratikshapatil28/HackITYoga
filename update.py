import numpy as np
import math

def accuracy(angle):
    # Step 1: Calculate the absolute difference
    diff = abs(angle - 65)
    #print(diff)
    # Step 2: Convert difference to error percentage
    err = (diff / 65) * 100
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

