#!/usr/bin/env python
# coding: utf-8

# In[6]:


import av
import numpy as np
from pydub import AudioSegment


# In[7]:


class AudioFrameHandler():
    def __init__(self,file_path: str = ''):
        self.custom_audio = AudioSegment.from_file(file_path,format='wav')
        self.len_audio = len(self.custom_audio)
        
        self.ms_per_audio_segment: int = 20
        self.audio_segm_shape: tuple
            
        self.play_state_tracker: dict = {'current_segment': -1}
        self.audio_segments_created: bool = False
        self.audio_segments: list = []
            
    def prepare_audio(self,frame: av.AudioFrame):
        raw_samples = frame.to_ndarray()
        sound = AudioSegment(
            data=raw_samples.tobytes(),
            sample_width = frame.format.bytes,
            frame_rate = frame.sample_rate,
            channels = len(frame.layout.channels)
            )
            
        self.ms_per_audio_segment = len(sound)
        self.audio_segm_shape = raw_samples.shape
            
        self.custom_audio = self.custom_audio.set_channels(sound.channels)
        self.custom_audio = self.custom_audio.set_frame_rate(sound.frame_rate)
        self.custom_audio = self.custom_audio.set_sample_width(sound.sample_width)
            
        self.audio_segments = [
                self.custom_audio[i: i + self.ms_per_audio_segment]
                for i in range (0, self.len_audio - self.len_audio % self.ms_per_audio_segment,self.ms_per_audio_segment)
            ]
            
        self.total_segments = len(self.audio_segments) - 1
        self.audio_segments_created = True
            
    def process(self,frame: av.AudioFrame,play_sound: bool = False):
        if not self.audio_segments_created:
            self.prepare_audio(frame)
        
        raw_samples = frame.to_ndarray()
        _current_segment = self.play_state_tracker['current_segment']
        
        if play_sound:
            if _current_segment < self.total_segments:
                _current_segment += 1
            else:
                _current_segment = 0
            
            sound = self.audio_segments[_current_segment]
            
        else:
            if -1 < _current_segment < self.total_segments:
                _current_segment += 1
                sound = self.audio_segments[_current_segment]
            else:
                _current_segment = -1
                sound = AudioSegment(
                    data=raw_samples.tobytes(),
                    sample_width=frame.format.bytes,
                    frame_rate=frame.sample_rate,
                    channels=len(frame.layout.channels),
                )
                sound = sound.apply_gain(-100)
        self.play_state_tracker['current_segment'] = _current_segment
        
        channel_sounds = sound.split_to_mono()
        channel_samples = [s.get_array_of_samples() for s in channel_sounds]
        
        new_samples = np.array(channel_samples).T
        
        new_samples = new_samples.reshape(self.audio_segm_shape)
        new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate

        return new_frame


# In[ ]:





# In[ ]:




