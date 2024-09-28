import cv2 as cv
import  mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing=mp.solutions.drawing_utils

drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

cap=cv.VideoCapture("WIN_20240926_19_41_38_Pro.mp4")

while cap.isOpened():
    success,image = cap.read()
    start=time.time()
    image=cv.resize(image,(800,760))
    image=cv.cvtColor(image,cv.COLOR_RGB2BGR)
    image=cv.cvtColor(cv.flip(image,1),cv.COLOR_BGR2RGB)

   
    #image.flags.writable=False
    results=face_mesh.process(image)
    #image.flags.writable=True
   

    img_h,img_w,img_c=image.shape
    face_3d=[]
    face_2d=[]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx,lm in enumerate(face_landmarks.landmark):
                if idx==33 or idx==263 or idx==1 or idx==61 or idx==291 or idx==199:
                    if idx==1:
                        nose_2d=(lm.x*img_w,lm.y*img_h)
                        nose_3d=(lm.x*img_w,lm.y*img_h,lm.z*3000)

                    x,y=int(lm.x*img_w),int(lm.y*img_h)
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])
            face_2d=np.array(face_2d,dtype=np.float64)
            face_3d=np.array(face_3d,dtype=np.float64)

            focal_length=1*img_w
            cam_matrix=np.array([[focal_length,0,img_h/2],
                                 [0,focal_length,img_w/2],
                                 [0,0,1]])
            dist_matrix=np.zeros((4,1),dtype=np.float64)
            success,rot_vec,trans_vec=cv.solvePnP(face_3d,face_2d,cam_matrix,dist_matrix)
            rmat,jac=cv.Rodrigues(rot_vec)
            angles,mtxR,mtxQ,Qx,Qy,Qz=cv.RQDecomp3x3(rmat)

            x=angles[0]*360
            y=angles[1]*360
            z=angles[2]*360

            
            if y< -6 and x>-1:
                text="left"
                if y<-6 and x>=7 :
                    text="left up "
            elif x<-1 and y<-2:
                text="left down"
            elif y> 4 and x>0 :
                text="right"
                if y>4 and x>5 :
                    text="right up"
            elif y>12 and  x<-1:
                text="right down"
            elif x<=0 and y>-4 :
                text="down" 
            elif x> 8:
                text= "up"
            else:
                text="forward"


            nose_3d_projection,jacobian=cv.projectPoints(nose_3d,rot_vec,trans_vec,cam_matrix,dist_matrix)

            p1=(int(nose_2d[0]),int(nose_2d[1]))
            p2=(int(nose_2d[0]+y*10),int(nose_2d[1]-x*10))

            cv.line(image,p1,p2,(255,0,0),3)

            cv.putText(image,text,(20,50),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            cv.putText(image,"x:"+str(np.round(x,2)),(500,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv.putText(image,"y:"+str(np.round(y,2)),(500,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv.putText(image,"z:"+str(np.round(z,2)),(500,150),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

      
       
    cv.imshow('head pose estimation',image)
    key = cv.waitKey(3)
    if key ==ord('q'):
        break


cap.release()



