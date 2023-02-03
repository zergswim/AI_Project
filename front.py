import streamlit as st

import io
import os
# import yaml

from PIL import Image
from PIL import ImageFont, ImageDraw

import cv2
import numpy as np

import requests

def main():
    st.title("Optical braille recognition Project")

    model = st.radio("Choose a model : ", ('yolo-n', 'yolo-m', 'yolo-x', 'retina'), horizontal=True)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        number = st.number_input('Confidence Threshold(0.0~0.9)', value=0.5, step=0.1, min_value=0.0, max_value=0.9)

        # st.image(image, caption='Uploaded Image')
        # st.write("Recognitioning...")
        st.write('The model is ', model)
        st.write('Confidence Threshold is ', number)

        params = {'score_limit': number}
        # print(params)
        file = [('file', (uploaded_file.name, image_bytes, uploaded_file.type))]
        if model == 'yolo-n':
            response = requests.post("http://localhost:30001/pred_yolon", files=file, params=params)
        elif model == 'yolo-m-best':
            response = requests.post("http://localhost:30001/pred_yolom", files=file, params=params)
        elif model == 'yolo-x':
            response = requests.post("http://localhost:30001/pred_yolox", files=file, params=params)
        else:
            response = requests.post("http://localhost:30001/pred_retina", files=file, params=params)

        boxes = response.json()["boxes"]
        labels = response.json()["labels"]

        lbl_type = st.radio("Choose label type : ", ('number', 'braille', 'english(test)'), horizontal=True)

        # st.write(f'label is {labels}')
        # st.write(f'boxes is {boxes}')

        img = np.array(image)
        for idx, box in enumerate(boxes):
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            lbl = labels[idx]
            # print(x1, y1, x2, y2, lbl, scr)
            # if scr > limit:
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)

            if lbl_type=="number":
                img = cv2.putText(img, str(lbl) , (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            elif lbl_type=="braille":
                pil_image = Image.fromarray(img)
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 20)
                draw = ImageDraw.Draw(pil_image)
                draw.text((int(x1), int(y1)), chr(10240+lbl), font=font, fill=(255,0,0,255))
                img = np.array(pil_image)
            else:
                pil_image = Image.fromarray(img)
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 18)
                draw = ImageDraw.Draw(pil_image)
                engchar = braille2engchar(chr(10240+lbl))
                draw.text((int(x1), int(y1)), engchar, font=font, fill=(0,0,0,255))
                img = np.array(pil_image)

        st.image(img, caption='Result')

def braille2engchar(bra):
    bra_dict = {'⠮':'the', '⠜':'ar', '⠣':'gh', '⠻':'er', '⠢':'en', '⠌':'st', '⠬':'ing', '⠿':'for',
                '⠯':'and', '⠔':'in', '⠲':'.', '⠠':'*', '⠫':'ed', '⠡':'ch', '⠪':'ow', '⠹':'th',
                '⠁':'a', '⠃':'b', '⠉':'c', '⠙':'d', '⠑':'e', '⠋':'f', '⠛':'g', 
                '⠓':'h', '⠊':'i', '⠚':'j', '⠅':'k', '⠇':'l', '⠍':'m', '⠝':'n',
                '⠕':'o', '⠏':'p', '⠟':'q', '⠗':'r', '⠎':'s', '⠞':'t', '⠥':'u', 
                '⠧':'v', '⠺':'w', '⠭':'x', '⠽':'y', '⠵':'z'
                }

    if bra in bra_dict.keys():
        rtn = bra_dict[bra]
    else:
        rtn = '?'

    return rtn 

main()

#streamlit run front.py --server.port 30002 --server.fileWatcherType none