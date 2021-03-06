import h5py
import tqdm
import json
import cv2
import os
from shutil import copyfile


def convert2coco_dataset(origin_dataset_path, target_folder, train_ratio=0.7):
    """target folder should have train,val and test folder"""

    mat_path = origin_dataset_path + 'train/train/digitStruct.mat'
    image_path = origin_dataset_path + 'train/train/'
    test_image_path = origin_dataset_path + 'test/test/'

    train_target_folder = target_folder + 'train/'
    val_target_folder = target_folder + 'val/'
    test_target_folder = target_folder + 'test/'

    coco_train = {
        'images': [],
        "annotations": [],
        "categories": [
            {
                'id': i,
                'name': str(i)
            } for i in range(10)
        ]
    }

    coco_val = {
        'images': [],
        "annotations": [],
        "categories": [
            {
                'id': i,
                'name': str(i)
            } for i in range(10)
        ]
    }

    coco_test = {
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
    train_length = int(length * train_ratio)
    for i in tqdm.tqdm(range(length)):
        # name
        name_ref = sheet['name'][i, 0]
        name = ''.join([chr(c[0]) for c in f[name_ref].value])

        if i < train_length:
            coco = coco_train
            copyfile(image_path + name, train_target_folder + name)
        else:
            coco = coco_val
            copyfile(image_path + name, val_target_folder + name)

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

    with open(target_folder + 'train.json', 'w') as fp:
        fp.write(json.dumps(coco_train, sort_keys=True, indent=4))

    with open(target_folder + 'val.json', 'w') as fp:
        fp.write(json.dumps(coco_val, sort_keys=True, indent=4))

    # move test set
    for filename in tqdm.tqdm(os.listdir(test_image_path)):
        img = cv2.imread(test_image_path + filename)
        image = {
            "id": int(filename.split('.')[0]),
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": filename,
        }
        coco_test['images'].append(image)
        copyfile(test_image_path + filename, test_target_folder + filename)

    # sort image order
    coco_test['images'].sort(key=lambda img: img['id'])

    with open(target_folder + 'test.json', 'w') as fp:
        fp.write(json.dumps(coco_test, sort_keys=True, indent=4))


def convert2yolo_dataset(origin_dataset_path, target_folder, train_ratio=0.7):
    """target folder should have train,val and test folder"""

    mat_path = origin_dataset_path + 'train/train/digitStruct.mat'
    image_path = origin_dataset_path + 'train/train/'
    test_image_path = origin_dataset_path + 'test/test/'

    train_image_folder = target_folder+'images/train/'
    val_image_folder = target_folder+'images/val/'
    test_image_folder = target_folder+'images/test/'

    train_label_folder = target_folder+'labels/train/'
    val_label_folder = target_folder+'labels/val/'

    f = h5py.File(mat_path)
    sheet = f['digitStruct']

    length = sheet['name'].shape[0]
    train_length = int(length * train_ratio)
    for i in tqdm.tqdm(range(length)):
        # name
        name_ref = sheet['name'][i, 0]
        name = ''.join([chr(c[0]) for c in f[name_ref].value])

        img = cv2.imread(image_path + name)
        img_h, img_w = img.shape[:2]

        if i < train_length:
            copyfile(image_path + name, train_image_folder + name)
            label_path = train_label_folder+name.replace('png', 'txt')
        else:
            copyfile(image_path + name, val_image_folder + name)
            label_path = val_label_folder+name.replace('png', 'txt')

        # bbox
        bbox_ref = sheet['bbox'][i, 0]
        bbox_length = f[bbox_ref]['label'].shape[0]

        if bbox_length > 0:
            with open(label_path, 'w') as fp:
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

                        fp.write(' '.join([
                                str(label),
                                str((left + width / 2) / img_w),
                                str((top + height / 2) / img_h),
                                str(width / img_w),
                                str(height / img_h)
                            ])+'\n'
                        )

                else:
                    label = int(f[bbox_ref]['label'][0, 0]) % 10
                    top = float(f[bbox_ref]['top'][0, 0])
                    left = float(f[bbox_ref]['left'][0, 0])
                    width = float(f[bbox_ref]['width'][0, 0])
                    height = float(f[bbox_ref]['height'][0, 0])

                    fp.write(' '.join([
                            str(label),
                            str((left + width / 2) / img_w),
                            str((top + height / 2) / img_h),
                            str(width / img_w),
                            str(height / img_h)
                        ]) + '\n'
                    )

    # move test set
    for filename in tqdm.tqdm(os.listdir(test_image_path)):
        copyfile(test_image_path + filename, test_image_folder + filename)


if __name__ == '__main__':
    convert2coco_dataset('../origin dataset/', './coco_data/')
    convert2yolo_dataset('../origin dataset/', './yolo_data/')
