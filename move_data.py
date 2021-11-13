import os
import tqdm
from shutil import copyfile

train_data_path = '../origin dataset/train/train/'
test_data_path = '../origin dataset/test/test/'

target_path = './data/'

# move test set
for filename in tqdm.tqdm(os.listdir(test_data_path)):
    copyfile(test_data_path+filename, target_path+'test/'+filename)

# move train set
origin_train_image_list = os.listdir(train_data_path)

length = len(origin_train_image_list)
train_length = int(length * 0.7)

i = 0
for filename in tqdm.tqdm(origin_train_image_list):
    if filename[-3:] != 'png':
        continue
    if i < train_length:
        copyfile(train_data_path+filename, target_path+'train/'+filename)
    else:
        copyfile(train_data_path + filename, target_path + 'val/' + filename)
    i += 1
