import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
import av
import cv2
import sys
import streamlit as st

from streamlit_webrtc import VideoTransformerBase,webrtc_streamer


st.title("Face Detection systme")
st.write("Maza Aaya")
cascPath =  "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

class VideoTransformer(VideoTransformerBase):
  def transform(self, frame):
    frame = frame.to_ndarray(format="bgr24")
    gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 5,minSize=(30,30))
    for (x,y,w,h) in faces:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    return frame
        
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
