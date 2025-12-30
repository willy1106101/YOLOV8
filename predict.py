from ultralytics import YOLO

# 載入模型（預訓練 or 自己訓練的）
model = YOLO("best.pt")
# model = YOLO("yolo11n.pt")

results = model.predict(
    source="images.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)

# 讀取結果
for r in results:
    print(r.boxes.xyxy)   # bounding boxes
    print(r.boxes.cls)    # class id
    print(r.boxes.conf)   # confidence
