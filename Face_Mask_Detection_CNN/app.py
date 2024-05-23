"""
important . create conda env with below python version sepearte req file for this face mask project
#conda create -p venv python==3.10.12
#conda activate [venv]
#conda install -r requirements.txt
#streamlit run app.py
"""
import os
_='''
2024-05-11 07:44:10.569217: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. 
You may see slightly different numerical results due to floating-point round-off errors from
 different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
'''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow import keras

st.title('Face Mask Detection (CNN Web App)')  

cnn = keras.models.load_model('Face_Mask_Detection_CNN.keras')

_='''
input_array = [[50, 50000.500, 8650,444000]]
input_array_np = np.array(input_array)
print(input_array_np)
print(type(input_array_np))
print(ann.predict(input_array_np)[0][0])
'''

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
  data = []
  image = Image.open(img_file_buffer)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)
  X = np.array(data)
  X_scaled = X/255
  print(X_scaled[0].shape)
  prediction = cnn.predict(X_scaled)
  print(prediction)
  if(prediction[0][0]>=prediction[0][1]):
    st.success("Customer is not wearing mask")
  else:
    st.success("Customer is wearing mask")  


#img_array = np.array(image)
_='''
if image is not None:
    st.image(
        image,
        caption=f"You amazing image has shape {img_array.shape[0:2]}",
        use_column_width=True,
    )
'''    