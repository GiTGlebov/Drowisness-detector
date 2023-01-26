#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import av
import threading
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from streamlit_jupyter import StreamlitPatcher, tqdm
from IPython import get_ipython
StreamlitPatcher().jupyter()
# import import_ipynb


# In[3]:


# %run eye_tracking.ipynb import VideoFrameHandler
from eye_tracking import VideoFrameHandler


# In[4]:


# %run Audio.ipynb import AudioFrameHandler
from Audio import AudioFrameHandler


# In[7]:


# if 'key' not in st.session_state:
#     st.session_state['key'] = 'value'

# st.session_state['key']


# In[8]:


alarm_file_path = 'C:/Users/glebk/Downloads/Erzhan_-_Vstavajj_63216378-_mp3cut.net_.wav'


st.set_page_config(
page_title='Drowsiness Detection',
layout='centered',
initial_sidebar_state='expanded',
menu_items={"About":'pet project'})

st.title('Drowsiness Detection!!')

col1, col2 = st.columns(spec=[1,1])

with col1:
    EAR_THRESH = st.slider("Eye aspect Ratio threshold:", 0.0,0.4,0.18,0.01)
    
with col2:
    WAIT_THRESH = st.slider("Second before sending alarm:", 0.0,0.5,1.0,0.25)
    

thresholds = {
    "EAR_THRESH" : EAR_THRESH,
    "WAIT_THRESH": WAIT_THRESH
}



video_handler = VideoFrameHandler()
audio_handler = AudioFrameHandler(file_path=alarm_file_path)

lock = threading.Lock()

shared_state = {"play_alarm": False}

def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="bgr24")
    frame,play_alarm = video_handler.main_process(frame,thresholds)
    with lock:
        shared_state['play_alarm'] = play_alarm
    return av.VideoFrame.from_ndarray(frame,format='bgr24')    

def audio_frame_callback(frame: av.AudioFrame):
    with lock:
        play_alarm = shared_state['play_alarm']
    new_frame: av.AudioFrame = audio_handler.process(frame,play_sound=play_alarm)
    return new_frame

# print(audio_frame_callback,video_frame_callback)
ctx = webrtc_streamer(
    key='driver_drowsy_detection',
    video_frame_callback=video_frame_callback,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"video": {"height": {"ideal": 480}}, "audio": True},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True,controls=False,muted=False),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


# In[ ]:




