import streamlit as st
from streamlit import caching
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import numpy as np
import shutil
import os
import cv2
import uuid 
################
#from inception import Inception
from vgg16 import VGG16
from utils import video_to_image
###################


### MENU VARIABLES ###
super_unsuper = ""
super_regre_class = ""
ML_option = ""
#######logo#########
image = Image.open('logo/yash_logo.png')
rgb_im = image.convert('RGB')

########################################
# Machine Learning Algorithms Menu
st.sidebar.image(rgb_im, use_column_width=False)
#st.sidebar.markdown("<h3 style='text-align: left; color: green;'>Model Selection</h3>", unsafe_allow_html=True)


#inception_obj = Inception()
vgg16_obj = VGG16()

########################################
# Title
st.markdown("<h2 style='text-align: center; color: black;'>Driver Destraction Identification</h2>", unsafe_allow_html=True)



uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	caching.clear_cache()
	image = Image.open(uploaded_file)
	id = uuid.uuid1() 
	file_name = str(id.hex)
	image_path = "images/{}.jpg".format(file_name)
	im1 = image.save(image_path) 
	st.image(image, caption='Uploaded Image.', use_column_width=True)
	vgg16_predict, _ = vgg16_obj.prediction(image_path)
	st.write("Prediction is: *{}*".format(vgg16_predict))

uploaded_file = st.file_uploader("Choose an video...", type="mp4")
if uploaded_file is not None:
	caching.clear_cache()
	id = uuid.uuid1() 
	file_name = str(id.hex)
	video_file_path = "videos/{}.mp4".format(file_name)
	fp = Path(video_file_path)
	fp.write_bytes(uploaded_file.getvalue())
	st.video(uploaded_file.getvalue(), format="video/mp4", start_time=0)
	#video_file_path = "videos/driver_distractor.mp4"
	image_paths = video_to_image(video_file_path)
	vgg16_predict = vgg16_obj.predict_video(image_paths)
	st.json(vgg16_predict)
	#st.write("VGG16 prediction is: *{}*".format(vgg16_predict))

	#st.video(data, format="video/mp4", start_time=0)
	