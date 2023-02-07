from fastapi import FastAPI, UploadFile, File
import uvicorn
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2
# # import requests
# from io import BytesIO

# FastAPI 객체 생성
app = FastAPI()

import torch
import torchvision
print(torchvision.__version__)

device = torch.device('cpu')

#YOLO v8
#pip install ultralytics
from ultralytics import YOLO

# "/"로 접근하면 return을 보여줌
@app.get("/")
def read_root():
    return "Optical braille recognition Project : /docs 로 API 테스트 가능합니다."
    # return {"Hello": "World1"}

# class Prod(BaseModel):
#     id: UUID = Field(default_factory=uuid4)
    # image_name: str
    # score_limit: Optional[float] = 0.5

# 레티나버전은 토치비전 업데이트(0.14) 필요예상 or fpn_v2 요구
# @app.post("/pred_retina", description="Retina v1 모델 예측")
# # async def get_pred(files: List[UploadFile] = File(...)):
# async def get_pred_retina(file: UploadFile = File(...), score_limit: Optional[float] = 0.5):

#     #Retina v1(torchvision)
#     # model_retina = torchvision.models.detection.retinanet_resnet50_fpn_v2(num_classes = 65, pretrained=False, pretrained_backbone = True)
#     model_retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 65, pretrained=False, pretrained_backbone = True)
#     model_retina.to(device)

#     checkpoint = torch.load('retina_v1_best.pth', map_location=device)
#     # state_dict = checkpoint.state_dict()
#     model_retina.load_state_dict(checkpoint)
#     model_retina.eval()
#     print("RetinaNet_v1 model loaded") 

#     # print(score_limit)
#     file.filename = f"./upload/{uuid4()}.jpg"
#     contents = await file.read() # <-- Important!

#     # example of how you can save the file
#     with open(file.filename, "wb") as f:
#         f.write(contents)

#     image = Image.open(file.filename)
#     image = np.array(image)
#     # print(img.shape)

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#     image /= 255.0
#     image = torch.tensor(image).permute(2,0,1)
#     image = image.unsqueeze(dim=0)
#     image = image.float().to(device)

#     outputs = model_retina(image)
#     boxes = outputs[0]['boxes'].int().detach().numpy().tolist()
#     scores = outputs[0]['scores'].detach().numpy().tolist()
#     labels = outputs[0]['labels'].detach().numpy().tolist()
#     rtn_boxes = []
#     rtn_labels = []

#     for idx, score in enumerate(scores):
#         if score > score_limit:
#             rtn_boxes.append(boxes[idx])
#             rtn_labels.append(labels[idx])

#     return {'boxes':rtn_boxes, 'labels':rtn_labels}

@app.post("/pred_yolon", description="YOLO v8 모델(n) 예측")
# async def get_pred(files: List[UploadFile] = File(...)):
async def get_pred_yolon(file: UploadFile = File(...), score_limit: Optional[float] = 0.5):

    # model_yolo = YOLO("yolov8n.yaml").to(device)
    model_yolo = YOLO("yolov8n_best.pt")
    model_yolo.to(device)
    print("YOLOn_v8 model loaded")

    # print(score_limit)
    file.filename = f"./upload/{uuid4()}.jpg"
    contents = await file.read() # <-- Important!

    # example of how you can save the file
    with open(file.filename, "wb") as f:
        f.write(contents)

    # print('file written')
        
    # image = plt.imread(file.filename)
    image = Image.open(file.filename)
    image = np.array(image)
    # print('shape:', image.shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # image /= 255.0
    # image = torch.tensor(image).permute(2,0,1)
    # image = image.unsqueeze(dim=0)
    # image = image.float().to(device)

    outputs = model_yolo.predict(source=image, device="cpu", visualize=False)
    # outputs = model_yolo(image)
    # print('outputs:', outputs)

    rtn_boxes = []
    rtn_labels = []
    
    # for output in outputs:
    boxes = outputs[0].boxes.xyxy.cpu().numpy().tolist()
    scores = outputs[0].boxes.conf.cpu().numpy().tolist()
    labels = outputs[0].boxes.cls.cpu().numpy().tolist()
    # print('output:', boxes, scores, labels)

    for idx, score in enumerate(scores):
        if score > score_limit:
            rtn_boxes.append(boxes[idx])
            rtn_labels.append(int(labels[idx]))

    return {'boxes':rtn_boxes, 'labels':rtn_labels}

@app.post("/pred_yolom", description="YOLO v8 모델(m) 예측")
async def get_pred_yolom(file: UploadFile = File(...), score_limit: Optional[float] = 0.5):

    # model_yolo = YOLO("yolov8m.yaml").to(device)
    model_yolo = YOLO("yolov8m_best.pt")
    model_yolo.to(device)
    print("YOLOm_v8 model loaded")

    file.filename = f"./upload/{uuid4()}.jpg"
    contents = await file.read()

    with open(file.filename, "wb") as f:
        f.write(contents)
        
    image = Image.open(file.filename)
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # image /= 255.0
    # image = torch.tensor(image).permute(2,0,1)
    # image = image.unsqueeze(dim=0)
    # image = image.float().to(device)

    outputs = model_yolo.predict(source=image, device="cpu", visualize=False)
    # outputs = model_yolo(image)
    # print('outputs:', outputs)

    rtn_boxes = []
    rtn_labels = []
    
    boxes = outputs[0].boxes.xyxy.cpu().numpy().tolist()
    scores = outputs[0].boxes.conf.cpu().numpy().tolist()
    labels = outputs[0].boxes.cls.cpu().numpy().tolist()

    for idx, score in enumerate(scores):
        if score > score_limit:
            rtn_boxes.append(boxes[idx])
            rtn_labels.append(int(labels[idx]))

    return {'boxes':rtn_boxes, 'labels':rtn_labels}

@app.post("/pred_yolox", description="YOLO v8 모델(x) 예측")
async def get_pred_yolox(file: UploadFile = File(...), score_limit: Optional[float] = 0.5):

    # model_yolo = YOLO("yolov8x.yaml").to(device)
    model_yolo = YOLO("yolov8x_best.pt")
    model_yolo.to(device)
    print("YOLOx_v8 model loaded")

    file.filename = f"./upload/{uuid4()}.jpg"
    contents = await file.read()

    with open(file.filename, "wb") as f:
        f.write(contents)
        
    image = Image.open(file.filename)
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # image /= 255.0
    # image = torch.tensor(image).permute(2,0,1)
    # image = image.unsqueeze(dim=0)
    # image = image.float().to(device)

    outputs = model_yolo.predict(source=image, device="cpu", visualize=False)
    # outputs = model_yolo(image)
    # print('outputs:', outputs)

    rtn_boxes = []
    rtn_labels = []
    
    boxes = outputs[0].boxes.xyxy.cpu().numpy().tolist()
    scores = outputs[0].boxes.conf.cpu().numpy().tolist()
    labels = outputs[0].boxes.cls.cpu().numpy().tolist()

    for idx, score in enumerate(scores):
        if score > score_limit:
            rtn_boxes.append(boxes[idx])
            rtn_labels.append(int(labels[idx]))

    return {'boxes':rtn_boxes, 'labels':rtn_labels}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=30001)
    # uvicorn.run(app, host="0.0.0.0", port=8000)

#python pred_api.py
#http://49.50.167.222:30001
#http://49.50.167.222:30001/docs
