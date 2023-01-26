#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import time
from IPython import get_ipython


# In[2]:


mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

denormalized_pixels = mp_drawing._normalized_to_pixel_coordinates

# %matplotlib inline


# In[3]:


all_left_eye_idx = list(mp_facemesh.FACEMESH_LEFT_EYE)
print(all_left_eye_idx)
all_left_eye_idx = set(np.ravel(all_left_eye_idx))
print(all_left_eye_idx)

all_right_eye_idx = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idx = set(np.ravel(all_right_eye_idx))

all_ind = all_left_eye_idx.union(all_right_eye_idx)


# In[4]:


chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]

all_chosen_eye_inxs = chosen_left_eye_idxs + chosen_right_eye_idxs


# In[5]:


img = cv2.imread('Gosling_dataset/Gosling_data/msg_129510079058_3Ry-gthumb-ghdata240.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(img)
img = np.ascontiguousarray(img)
print(img)
img_H,img_W,_ = img.shape

plt.imshow(img)


# In[6]:


with mp_facemesh.FaceMesh(static_image_mode=True,
                          max_num_faces=1,
                          refine_landmarks=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                         ) as face_mesh:
    results = face_mesh.process(img)

print(bool(results.multi_face_landmarks))    


# In[7]:


landmark0 = results.multi_face_landmarks[0].landmark[0]
print(landmark0)

landmark0_x = landmark0.x * img_W
landmark0_y = landmark0.y * img_H
landmark0_z = landmark0.z * img_W
print()

print(landmark0_x,landmark0_y,landmark0_z, sep = '\n')
print()

print(results.multi_face_landmarks[0].landmark)


# In[8]:


def plot(img_W,img_H,
        img_dt,img_eye_lmks=None,
        img_eye_chosen_lmks=None,
        face_landmarks=None,
        ts_thickness=1,
        ts_circle_radius=2,
        lmk_circle_radius=3,
        name='1',
        ):
    image_drawing_tool=img_dt
    image_eye_lmks = img_dt.copy() if img_eye_lmks is None else img_eye_lmks
    image_eye_chosen_lmks = img_dt.copy() if img_eye_chosen_lmks is None else img_eye_chosen_lmks
    
    connections_drawing_spec = mp_drawing.DrawingSpec(
    thickness=ts_thickness,
    circle_radius=ts_circle_radius,
    color=(255,255,255))
    
    fig = plt.figure(figsize=(20, 15))
    fig.set_facecolor("white")
    mp_drawing.draw_landmarks(image=image_drawing_tool,
                             landmark_list=face_landmarks,
                             connections=mp_facemesh.FACEMESH_TESSELATION,
                             landmark_drawing_spec=None,
                             connection_drawing_spec=connections_drawing_spec)
    
    landmarks = face_landmarks.landmark
    
    for landmark_idx,landmark in enumerate(landmarks):
        if landmark_idx in all_ind:
            pred_cord = denormalized_pixels(landmark.x,landmark.y,img_W,img_H)
#             print(image_eye_lmks,pred_cord,lmk_circle_radius, sep='\n')
            cv2.circle(image_eye_lmks,pred_cord,lmk_circle_radius,(255,255,255),-1)
        if landmark_idx in all_chosen_eye_inxs:
            pred_cord = denormalized_pixels(landmark.x,landmark.y,img_W,img_H)
            cv2.circle(image_eye_chosen_lmks,pred_cord,lmk_circle_radius,(255,255,255),-1)
            
    plt.subplot(1,3,1)
    plt.title("Face Tesselation",fontsize=18)
    plt.imshow(image_drawing_tool)
    plt.axis('off')
    
    plt.subplot(1,3,2)
    plt.title("All eye landmarks",fontsize=18)
    plt.imshow(image_eye_lmks)
    plt.axis('off')
    
    plt.subplot(1,3,3)
    plt.title("Chosen eye landmarks",fontsize=18)
    plt.imshow(image_eye_chosen_lmks)
    plt.axis('off')
    
    plt.show()
    plt.close()
    return
    


# In[9]:


if results.multi_face_landmarks:
    for face_id,face_landmarks in enumerate(results.multi_face_landmarks):
        _=plot(img_W=img_W,img_H=img_H,img_dt=img.copy(),ts_circle_radius=0.5,ts_thickness=1,lmk_circle_radius=1,face_landmarks=face_landmarks)


# In[10]:


def distance(p1,p2):
    dist = sum([(i-j)**2 for i,j in zip(p1,p2)]) ** 0.5
    return dist

def get_ear_func(landmarks,refer_idxs,frame_w,frame_h):
    try:
        
        coord_point = []
        for i in refer_idxs:
            lmks = landmarks[i]
            coord = denormalized_pixels(lmks.x,lmks.y,frame_w,frame_h)
            coord_point.append(coord)
            
        P2_P6 = distance(coord_point[1],coord_point[5])
        P3_P5 = distance(coord_point[2],coord_point[4])
        P1_P4 = distance(coord_point[0],coord_point[3])
        
        ear = (P2_P6 + P3_P5) / (2.0*P1_P4)
    except:
        ear = 0
        coord_point = None
    return ear,coord_point 


# In[11]:


def calc_avg_ear(landmarks,left_eye_idxs,right_eye_idxs,image_w,image_h):
    left_ear,left_lmks_coord = get_ear_func(landmarks,left_eye_idxs,image_w,image_h)
#     print(left_ear)
    
    right_ear, right_lmks_coord = get_ear_func(landmarks,right_eye_idxs,image_w,image_h)
#     print(right_ear)
    AVG_EAR = (left_ear + right_ear) / 2.0
    return AVG_EAR,(left_lmks_coord,right_lmks_coord)


# In[12]:


img_close = cv2.imread('C:/Users/glebk/Downloads/0dc36e3d0fc3e942251ff99672ee2df6.jpg')[:,:,::-1]
img_open = cv2.imread('Gosling_dataset/Gosling_data/HQ_00029-gthumb-ghdata240.jpg')[:,:,::-1]

for idx,image in enumerate([img_open,img_close]):
    image = np.ascontiguousarray(image)
#     print(idx)
    imgH,imgW,_ = image.shape
#     print(image.shape)
    
    custom_chosen_lmnk_img = image.copy()
    with mp_facemesh.FaceMesh(refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image).multi_face_landmarks
#         print(results)
        if results:
            for face_id,face_lmks in enumerate(results):
                landmarks = face_lmks.landmark
                
                EAR,_ = calc_avg_ear(landmarks,chosen_left_eye_idxs,chosen_right_eye_idxs,
                                    imgW,imgH)
                pos_text = (1,int(imgH/10))
                size_text = 3 if (imgW>=600 and imgH>=600) else 0.9 
                cv2.putText(custom_chosen_lmnk_img,
                           f"EAR{round(EAR,2)}",pos_text,
                           cv2.FONT_HERSHEY_COMPLEX,
                           size_text,(255,255,255),2)
                ts_circle_radius = 5 if (imgW>=600 and imgH>=600) else 1 
                lmk_circle_radius = 5 if (imgW>=600 and imgH>=600) else 1 
                plot(img_W=imgW,
                     img_H=imgH,
                    img_dt=image.copy(),
                    img_eye_chosen_lmks=custom_chosen_lmnk_img,
                    face_landmarks=face_lmks,
                    ts_thickness=1,
                    ts_circle_radius=ts_circle_radius,
                    lmk_circle_radius=lmk_circle_radius)


# In[13]:


def get_mediapipe_app(max_num_faces=1,
                      refline_landmarks=True,
                      min_detection_score=0.5,
                      min_tracking_score=0.5):
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_num_faces,
                                   refine_landmarks=refline_landmarks,
                                   min_detection_confidence=min_detection_score,
                                   min_tracking_confidence=min_tracking_score)
    return face_mesh


# In[14]:


def plot_eye_landmarks(frame,
                      left_lm_coord,
                      right_lm_coor,
                      colors):
    for lm_coord in [left_lm_coord,right_lm_coor]:
        if lm_coord:
            for coord in lm_coord:
                cv2.circle(frame,coord,2,colors,-1)
    frame = cv2.flip(frame,1)       
    return frame
                    


# In[15]:


def plot_text(image,text,origin,color,font=cv2.FONT_HERSHEY_COMPLEX,fnt_scale=0.8,thickness=2):
    image = cv2.putText(image,text,origin,font,fnt_scale,color,thickness)
    return image


# In[16]:


class VideoFrameHandler:
    def __init__(self):
        self.eye_indxs = {
            'left': [362, 385, 387, 263, 373, 380],
            'right': [33, 160, 158, 133, 153, 144]
        }
        
        self.Red = (0,0,255)
        self.Green = (0,255,0)
        
        self.face_mesh_model = get_mediapipe_app()
        
        self.state_tracker = {
            'start_time': time.perf_counter(),
            "Drowsy_time": 0.0,
            "Color": self.Green,
            "Play_ALARM": False
        }
            
        self.EAR_txt_pos = (10,30)
    def main_process(self, frame:np.array, treshholds:dict):
        frame.flags.writeable = False
        frame_H, frame_W,_ = frame.shape
        
        Drowsy_time_txt_pos = (10,int(frame_H // 2 *  1.7))
        ALARM_txt_pos = (10,int(frame_H // 2 *  1.85))
        
        results = self.face_mesh_model.process(frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            EAR,coord = calc_avg_ear(landmarks,
                                     self.eye_indxs['left'],
                                     self.eye_indxs['right'],
                                     frame_W,frame_H
                                    )
            
            frame = plot_eye_landmarks(frame,
                                      coord[0],
                                      coord[1],
                                      self.state_tracker['Color'])
            
            if EAR < treshholds["EAR_THRESH"]:
#                 print("ALARM")
                end_time = time.perf_counter()
                print(end_time)
                self.state_tracker['Drowsy_time'] += end_time - self.state_tracker['start_time']
#                 print(self.state_tracker['Drowsy_time'])
                self.state_tracker['start_time'] = end_time
#                 print(self.state_tracker['start_time'])
                self.state_tracker['Color'] = self.Red
                
                if self.state_tracker["Drowsy_time"] > treshholds["WAIT_THRESH"]:
                    self.state_tracker["Play_ALARM"] = True
                    plot_text(frame,"Wake UP!!!",ALARM_txt_pos,self.state_tracker["Color"])
                    
                    
            else:
                self.state_tracker['start_time'] = time.perf_counter()
                self.state_tracker['Drowsy_time'] = 0.0
                self.state_tracker['Color'] = self.Green
                self.state_tracker["Play_ALARM"] = False
                    
            EAR_txt = f"EAR: {round(EAR,2)}"
            Drowsy_time_txt = f"Drowsy_time: {round(self.state_tracker['Drowsy_time'],3)} secs"
                
            plot_text(frame,EAR_txt,self.EAR_txt_pos,self.state_tracker["Color"])
            plot_text(frame,Drowsy_time_txt,Drowsy_time_txt_pos,self.state_tracker["Color"])
                
        else:
            self.state_tracker['Drowsy_time'] = 0.0
            self.state_tracker['start_time'] = time.perf_counter()
            self.state_tracker["Color"] = self.Green
            self.state_tracker['Play_ALARM'] = False
                
#                 frame = cv2.flip(frame,1)
                
        return frame,self.state_tracker['Play_ALARM']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




