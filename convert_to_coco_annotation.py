import h5py
import tqdm
import json
import cv2


def mat_to_coco_format(mat_path, image_path, save_path):
    coco = {
        'images': [],
        "annotations": [],
        "categories": [
            {
                'id': i,
                'name': str(i)
            } for i in range(10)
        ]
    }

    f = h5py.File(mat_path)
    sheet = f['digitStruct']

    length = sheet['name'].shape[0]
    for i in tqdm.tqdm(range(length)):
        # name
        name_ref = sheet['name'][i, 0]
        name = ''.join([chr(c[0]) for c in f[name_ref].value])

        # add image
        img = cv2.imread(image_path+name)
        image = {
            "id": i,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": name,
        }
        coco['images'].append(image)

        # bbox
        bbox_ref = sheet['bbox'][i, 0]
        bbox_length = f[bbox_ref]['label'].shape[0]

        if bbox_length > 1:
            for j in range(bbox_length):
                label_ref = f[bbox_ref]['label'][j, 0]
                top_ref = f[bbox_ref]['top'][j, 0]
                left_ref = f[bbox_ref]['left'][j, 0]
                width_ref = f[bbox_ref]['width'][j, 0]
                height_ref = f[bbox_ref]['height'][j, 0]

                label = int(f[label_ref].value[0, 0]) % 10
                top = float(f[top_ref].value[0, 0])
                left = float(f[left_ref].value[0, 0])
                width = float(f[width_ref].value[0, 0])
                height = float(f[height_ref].value[0, 0])

                annotation = {
                    "id": len(coco['annotations']),
                    "image_id": i,
                    "category_id": label,
                    "segmentation": [],
                    "area": height * width,
                    "bbox": [int(left), int(top), int(width), int(height)],
                    "iscrowd": 0,
                }

                coco['annotations'].append(annotation)
        else:
            label = int(f[bbox_ref]['label'][0, 0]) % 10
            top = float(f[bbox_ref]['top'][0, 0])
            left = float(f[bbox_ref]['left'][0, 0])
            width = float(f[bbox_ref]['width'][0, 0])
            height = float(f[bbox_ref]['height'][0, 0])

            annotation = {
                "id": len(coco['annotations']),
                "image_id": i,
                "category_id": label,
                "segmentation": [],
                "area": height * width,
                "bbox": [int(left), int(top), int(width), int(height)],
                "iscrowd": 0,
            }

            coco['annotations'].append(annotation)

    with open(save_path, 'w') as fp:
        fp.write(json.dumps(coco, sort_keys=True, indent=4))


mat_path = '../origin dataset/train/train/digitStruct.mat'
img_path = '../origin dataset/train/train/'
save_path = './data/COCOAnnotation.json'
mat_to_coco_format(mat_path, img_path, save_path)
