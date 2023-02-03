#pip install ultralytics

from ultralytics import YOLO

# Load a model0
model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# model = YOLO("yolov8x.yaml")  # build a new model from scratch
# model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# model = YOLO("/opt/ml/fastapi/yolov8m_best.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/opt/ml/runs/detect/train20/weights/last.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/opt/ml/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/opt/ml/runs/detect/train9/weights/best.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/opt/ml/runs/detect/train6/weights/best.pt")  # load a pretrained model (recommended for training)

# # Use the model
# results = model.train(data="coco128.yaml", epochs=1)  # train the model
# results = model.train(data="/opt/ml/DATA_YOLO/dataset.yaml", epochs=100, fliplr=0.0, translate=0.0, scale=0.0)  # train the model
# results = model.train(data="/opt/ml/DATA_YOLON/dataset.yaml", epochs=100, fliplr=0.0, translate=0.0)
# results = model.train(data="/opt/ml/DATA_YOLON2/dataset.yaml", epochs=100, imgsz=1024, workers=2, batch=6, fliplr=0.0, translate=0.0, image_weights=True)  
# results = model.train(data="/opt/ml/DATA_YOLON2/dataset.yaml", epochs=100, imgsz=720, workers=2, batch=10, fliplr=0.0, translate=0.0, image_weights=True, mosaic=0.0)  
# results = model.train(data="/opt/ml/DATA_YOLON2/dataset.yaml", epochs=1, imgsz=1024, workers=2, batch=12, fliplr=0.0, close_mosaic=10, translate=0.1, image_weights=True, visualize=False, optimizer="AdamW", cos_lr=False)  
#
# results = model.train(data="/opt/ml/DATA_YOLON3/dataset.yaml", epochs=100, imgsz=640, workers=2, batch=16, fliplr=0.0, close_mosaic=10, translate=0.1, image_weights=True, visualize=False, optimizer="AdamW", cos_lr=False, scale=0.8)

# results = model.train(data="/opt/ml/DATA_YOLON3/dataset.yaml", epochs=100, imgsz=800, workers=2, batch=12, fliplr=0.0, close_mosaic=10, translate=0.1, image_weights=True, visualize=False, optimizer="AdamW", cos_lr=False, scale=0.8)

# results = model.train(data="/opt/ml/DATA_YOLON2/dataset.yaml", epochs=50, imgsz=912, workers=2, batch=12, fliplr=0.0, close_mosaic=15, translate=0.2, image_weights=True, visualize=False, cos_lr=False, scale=0.8, max_det=400, perspective=0.1, degrees=0.1)

results = model.train(data="/opt/ml/DATA_YOLON2/dataset.yaml", epochs=60, imgsz=800, workers=2, batch=12, fliplr=0.0, close_mosaic=10, translate=0.2, image_weights=True, visualize=False, cos_lr=False, scale=0.8, max_det=400)

# results = model.train(data="/opt/ml/DATA_YOLON2/dataset.yaml", epochs=40, imgsz=800, workers=2, batch=12, fliplr=0.0, close_mosaic=10, translate=0.1, image_weights=True, visualize=False, optimizer="AdamW", cos_lr=False, scale=0.8, lr0=0.00005, max_det=500)

#, resume="/opt/ml/fastapi/yolov8m_last.pt")
model.val() 
# results = model.train(data="./datasets/coco128", epochs=1)  # train the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format

# results = model('https://test.narangdesign.com/mail/kbuwel/202011/images/news_01_img1.jpg', save=True, save_txt=True)  # predict on an image
# print(results)

# model.val()
success = model.export(format="onnx")  # export the model to ONNX format