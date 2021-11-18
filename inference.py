import torch
import os
import json
import tqdm
from PIL import Image

model_path = './yolov5/runs/train/exp3/weights/best.pt'
model = torch.hub.load('./yolov5', 'custom',
                       path=model_path, source='local')

test_path = 'yolo_data/images/test/'
img_list = os.listdir(test_path)
img_list.sort(key=lambda x: int(x[:-4]))

answer = []

for img_name in tqdm.tqdm(img_list):
    img = Image.open(test_path+img_name)
    result = model(img, size=640)

    bboxes = result.pandas().xyxy[0]
    for index, row in bboxes.iterrows():
        left = row['xmin']
        top = row['ymin']
        width = row['xmax'] - row['xmin']
        height = row['ymax'] - row['ymin']
        score = row['confidence']
        label = row['class']

        b = {
            "image_id": int(img_name[:-4]),
            "bbox": [left, top, width, height],
            "score": score, "category_id": label
        }

        answer.append(b)

with open('answer.json', 'w') as fp:
    fp.write(json.dumps(answer))
