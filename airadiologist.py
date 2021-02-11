import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

st.sidebar.header('Examples of scans')
st.write('<small style="background-color:Red; text-align:center;">\
**This is not a diagnostic tool and is not intended to be used as such**\
</small>', unsafe_allow_html=True)

st.sidebar.write('&nbsp;')
st.sidebar.subheader('Normal')
st.sidebar.image(['./data/NORMAL/NORMAL (27).png',
'./data/NORMAL/NORMAL (8).png'])

st.sidebar.write('&nbsp;')
st.sidebar.subheader('Viral Pneumonia')
st.sidebar.image(['data\Viral Pneumonia\Viral Pneumonia (50).png',
'data\Viral Pneumonia\Viral Pneumonia (254).png'])

st.sidebar.write('&nbsp;')
st.sidebar.subheader('Covid-19')
st.sidebar.image(['data\COVID\COVID (236).png',
'data\COVID\COVID (361).png'])

background = \
'''
    <style>
    body \
    {
        background-image: url('https://static01.nyt.com/images/2017/02/16/well/doctors-hospital-design/doctors-hospital-design-jumbo.jpg');
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-size: 100% 100%;
    }
    </style>
'''

st.markdown(background, unsafe_allow_html=True)

def covidtester(testimage):
    model = load_model('covidpredictor.h5')
    image = img_to_array(ImageOps.fit(testimage, (256, 256)))
    image = image.reshape((1, 256, 256, 1))
    prediction = model.predict_classes(image)[0]
    percent = round(model.predict_proba(image)[0][prediction]*100,2)
    return prediction, percent

st.write('<h1 style="color:White; text-align:center;">\
How cAn I help you?</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Import your image here", type=["jpg", "png"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Successful', use_column_width=True)
        st.write("")

        st.write("Asking your image to turn and cough")
        pred, perc = covidtester(image)
        if pred == 0:
            st.write('<p style="color:White; background-color:Red; text-align:center;">\
            You likely have Covid</p>', unsafe_allow_html=True)
            st.write(':mask:')
        elif pred == 1:
            st.write('<p style="background-color:Green; text-align:center;">\
            You are healthy!</p>', unsafe_allow_html=True)
            st.write(':sunglasses:')
        elif pred == 2:
            st.write('<p style="background-color:Yellow; text-align:center;">\
            You likely have pneumonia</p>', unsafe_allow_html=True)
            st.write(':sneezing_face:')

        st.write(f'We are {perc}% sure!')
        st.write('<p style="color:SaddleBrown; background-color:Gold; text-align:center;">\
        (NOTE: despite high confidences it is impossible to be 100% confident in any prediction)\
        </p>', unsafe_allow_html=True)
