import cv2 as cv
import numpy as np
import mediapipe as mp
import math
mp_face_mesh = mp.solutions.face_mesh
RIGHT_IRIS=[474,475,476,477]
LEFT_IRIS=[469,470,471,472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]
L_H_UP = [69]
L_H_DOWN = [101]
R_H_DOWN = [299]
R_H_UP = [330]
L_H_CENTER = [468]
R_H_CENTER = [473]

def euclidean_distance(point1,point2,):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance
def iris_position(iris_center,right_point,left_point,):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point,left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio<=0.42:
        iris_position="left"
    elif ratio>0.42 and ratio<=0.66:
        iris_position = "center"
    elif ratio>0.66:
        iris_position="right"
   
    return iris_position, ratio

def iris_vertical_position(iris_center, upper_point, lower_point):
    up_dist = euclidean_distance(iris_center, upper_point)
    down_dist = euclidean_distance(iris_center, lower_point)
    total_dist = up_dist+down_dist
    dist= up_dist / total_dist
    iris_vertical_position=""
    
    if dist >0.49:
        iris_vertical_position = "up"
    elif dist>=0.44 and dist<= 0.49:
        iris_vertical_position = "center"
    elif dist<0.44:
        iris_vertical_position = "down"
    
    return iris_vertical_position,dist

     
cap= cv.VideoCapture("5442623-uhd_3840_2160_25fps.mp4")
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret,frame=cap.read()
        frame = cv.resize(frame,(800,760))  
        if cv.error:
            break
        if not ret or frame is None:
            break
        frame=cv.flip(frame,1)
        rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        img_h,img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
           
           mesh_points=np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
           
           (l_cx,l_cy),l_radius= cv.minEnclosingCircle(mesh_points[L_H_CENTER])
           (r_cx,r_cy),r_radius= cv.minEnclosingCircle(mesh_points[R_H_CENTER])

           center_left=np.array([l_cx,l_cy],dtype=np.int32)
           center_right=np.array([r_cx,r_cy],dtype=np.int32)

           cv.circle(frame,mesh_points[R_H_CENTER][0],3,(255,255,255),1,cv.LINE_AA)
           cv.circle(frame,mesh_points[L_H_CENTER][0],3,(255,255,255),1,cv.LINE_AA)

           cv.circle(frame,mesh_points[R_H_RIGHT][0],3,(255,255,255),-1,cv.LINE_AA)
           cv.circle(frame,mesh_points[R_H_LEFT][0],3,(0,255,255),-1,cv.LINE_AA)
           cv.circle(frame,mesh_points[L_H_RIGHT][0],3,(255,255,255),-1,cv.LINE_AA)
           cv.circle(frame,mesh_points[L_H_LEFT][0],3,(0,255,255),-1,cv.LINE_AA)
           cv.circle(frame,mesh_points[R_H_UP][0],3,(0,255,255),-1,cv.LINE_AA)
           cv.circle(frame,mesh_points[R_H_DOWN][0],3,(0,255,255),-1,cv.LINE_AA)
           cv.circle(frame, mesh_points[L_H_DOWN][0], 3, (0, 255, 255), -1, cv.LINE_AA)
           cv.circle(frame, mesh_points[L_H_UP][0], 3, (0, 255, 255), -1, cv.LINE_AA)
           
           

           
           iris_pos,ratio=iris_position(center_right, mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0])
           iris_pos,ratio=iris_position(center_left, mesh_points[L_H_LEFT],mesh_points[L_H_RIGHT][0])
          
           up_dist,dist= iris_vertical_position(center_right, mesh_points[R_H_UP], mesh_points[R_H_DOWN][0])
           
           
           cv.putText(frame,f"Iris pos: {iris_pos} {ratio:.2f}",(30,30),cv.FONT_HERSHEY_PLAIN,1.2,(0,0,0),1,cv.LINE_AA,)
           cv.putText(frame,f"eye: {up_dist}{dist:.2f}",(30,50),cv.FONT_HERSHEY_PLAIN,1.2,(0,0,0),1,cv.LINE_AA)
           


           cv.imshow('img',frame)
        key = cv.waitKey(3)
        if key ==ord('q'):
            break

cap.release()
cv.destroyAllWindows()
