'''
The file convert both image and labels into tfrecords files
Label is parsed from the corresponding xml file
'''
import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
import scipy.io as sio
from glob import glob
import multiprocessing
from joblib import Parallel, delayed

raw_data_path = '../data/raw'
tfrecords_data_path = '../data/tfrecords'
tag_file = '../data/17tags_meta.txt'
tags_meta = []
with open(tag_file) as f:
    tags_meta = f.read().lower().split('\n')
NUM_TAGS = len(tags_meta)

print("NUM_TAGS:",NUM_TAGS)
# tags_meta is a list of tuples, each element is a tuples, like (0, 'bcc')
tags_meta = zip(np.arange(0, NUM_TAGS), tags_meta)

TRAIN_RATIO = 0.8
IMAGESIZE = 256

# data_file_name could be like ../../data/SyntheticScatteringData/014267be_varied_sm/00000001.mat
# ../../data/SyntheticScatteringData/014267be_varied_sm/analysis/results/00000001.xml
# return a label vector of size [num_of_tags]
def parse_label(data_file_name):
    data_path = data_file_name.rsplit('/',1)
    print("data_file_name:",data_file_name, "data_path:",data_path)
    label_file_name = data_path[0] + '/analysis/results/' + data_path[1].split('.')[0] + '.xml'
    print("label_file_name:", label_file_name)
    label_vector = np.zeros([NUM_TAGS])
    

    if os.path.exists(label_file_name):
        root = ET.parse(label_file_name).getroot()
        res_count = 0
        for result in root[0]:
            res_count+=1
            # print("res_count:",res_count)
            attribute = result.attrib.get('name')
            # print("attribute1:",attribute)
            attribute = attribute.rsplit('.', 1)
            # print("attribute2:",attribute)
            # only care about high level tags
            attribute = attribute[1].split(':')[0]
            print("attribute3:",attribute)
            # check if that attribute is with the all_tag_file
            for i in range(NUM_TAGS):
                # if tags_meta[i][1] == attribute:
                if tags_meta[i][1].lower() == attribute.lower():
                    label_vector[tags_meta[i][0]] = 1
                    print("label_vector:",label_vector ) 
    else:
        print ('%s does not exist!' %label_file_name)

    flag = bool(sum(label_vector))
    print("flag:",flag, label_vector)
    return label_vector, flag

# helper function for binary data
# return a string of binary files about all files contain in that directory
# store label first, then image
# they are both encoded in float64 (double) format
def _binaryize_one_dir(dir):
    file_names = os.listdir(dir)
    string_binary = ''
    i = 0
    for data_file in file_names:
        i+=1
        if os.path.isfile(os.path.join(dir, data_file)):
            # if i >2:
            #     break
            
            label, flag = parse_label(os.path.join(dir, data_file))
            print("dir:",dir)
            print("label:",label,"flag:",flag)
            if not flag:
                print(os.path.join(dir, data_file))
            else:
                label = label.astype('int16')
                label = list(label)
                label_byte = struct.pack('h'*len(label), *label) #?? 
                string_binary += label_byte
                image = sio.loadmat(os.path.join(dir, data_file))
                # the shape of image is 256*256
                image = image['detector_image']
                
                print("image:",image.shape)
                print("label:",label)

                image = np.reshape(image, [-1])
                
                # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0
                image = image.astype('int16')
                
                image = list(image)
                
                image_byte = struct.pack('h'*len(image), *image)
                string_binary += image_byte

    return string_binary

### modified from the above function "_binaryize_one_dir" to get image and labels only
def getImageLabel(dir):
    file_names = os.listdir(dir)
    string_binary = ''
    imageTensor = []
    labelTensor = []
    i = 0
    for data_file in file_names:
        i+=1
        if os.path.isfile(os.path.join(dir, data_file)):
#             if i >2:
#                 break
            print("number of data_file:",i)
            label, flag = parse_label(os.path.join(dir, data_file))
            print("dir:",dir)
            print("label:",label,"flag:",flag)
            if not flag:
                print(os.path.join(dir, data_file))
            else:
                label = label.astype('int16')
                # label = list(label)
                # label_byte = struct.pack('h'*len(label), *label)
                # string_binary += label_byte
                image = sio.loadmat(os.path.join(dir, data_file))
                # the shape of image is 256*256
                image = image['detector_image']
                
                # print("image:",image)
                # print("label:",label)
                # image = np.reshape(image, [-1])
                # # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0
                image = image.astype('int16')
                # image = list(image)
                # image_byte = struct.pack('h'*len(image), *image)
                # string_binary += image_byte
                imageTensor.append(image)
                labelTensor.append(label)
    
    #return string_binary
    return np.array(imageTensor), np.array(labelTensor)

def _get_one_binary_file(dirs, save_name, i):
    print('processing %s_%d.bin' % (save_name, i))
    with open(os.path.join(tfrecords_data_path, '%s_%d.bin' % (save_name, i)), 'wb') as f:
        for dir_list_i in dirs:
            print("dir_list_i:",dir_list_i)
            f.write(_binaryize_one_dir(dir_list_i))

def _get_one_tensor_object(dirs):
    '''modified from the above function "_get_one_binary_file" '''
    #with open(os.path.join(tfrecords_data_path, '%s_%d.bin' % (save_name, i)), 'wb') as f:
    
    count = 0
    for dir_list_i in dirs:
        
        print("")
        print("dir_list_i:",dir_list_i)
        if count == 0:
            imageTensorMerged, labelTensorMerged = getImageLabel(dir_list_i)
            # print("imageTensorMerged0.shape:",imageTensorMerged.shape)
            # print("labelTensorMerged0.shape:",labelTensorMerged.shape)
        else:
            imageTensorTemp, labelTensorTemp= getImageLabel(dir_list_i)
            imageTensorMerged = np.concatenate((imageTensorMerged,imageTensorTemp),axis =0)
            labelTensorMerged = np.concatenate((labelTensorMerged,labelTensorTemp),axis =0)
        count +=1
    print("imageTensorMerged.shape:",imageTensorMerged.shape)
    print("labelTensorMerged.shape:",labelTensorMerged.shape)
    return imageTensorMerged ,labelTensorMerged 


def processFile():
    # dirs = os.listdir(DATA_PATH)
    dirs = glob(raw_data_path+'/*')
    length_dirs = len(dirs)
    num_dirs_per_bin = 50
    idx = np.random.permutation(len(dirs))
    num_bin_file = int(np.ceil(len(dirs) / num_dirs_per_bin))
    num_train_bin_file = int(TRAIN_RATIO * num_bin_file)
    num_val_bin_file = num_bin_file - num_train_bin_file

    # get training data
    train_dirs = []
    for i in range(num_train_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range(i*num_dirs_per_bin, (i+1)*num_dirs_per_bin)]
        train_dirs.append(tmp_dir)

    # get val data
    val_dirs = []
    for i in range(num_val_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range((i+num_train_bin_file)*num_dirs_per_bin,  (i+1+num_train_bin_file)*num_dirs_per_bin)]
        val_dirs.append(tmp_dir)

    if not os.path.exists(tfrecords_data_path):
        os.mkdir(tfrecords_data_path)
    print(num_train_bin_file)
    print(num_val_bin_file)

    num_cores = multiprocessing.cpu_count()/2
    Parallel(n_jobs=num_cores)(
        delayed(_get_one_binary_file)(train_dirs[i], 'train_batch', i) for i in range(num_train_bin_file))

    Parallel(n_jobs=num_cores)(
        delayed(_get_one_binary_file)(val_dirs[i], 'val_batch', i) for i in range(num_val_bin_file))

    
    print("length_dir:",length_dirs)


def processTensor():
    ''' modified from the above function '''
    # dirs = os.listdir(DATA_PATH)
    dirs = glob(raw_data_path+'/*')
    length_dirs = len(dirs)
    num_dirs_per_bin = length_dirs

    idx = np.random.permutation(len(dirs))
    num_bin_file = int(np.ceil(len(dirs) / num_dirs_per_bin))
    # num_train_bin_file = int(TRAIN_RATIO * num_bin_file)
    # num_val_bin_file = num_bin_file - num_train_bin_file

    # get training data
    train_dirs = []
    #for i in range(num_train_bin_file):
    for i in range(num_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range(i*num_dirs_per_bin, (i+1)*num_dirs_per_bin)]
        train_dirs.append(tmp_dir)

    # # get val data
    # val_dirs = []
    # for i in range(num_val_bin_file):
    #     tmp_dir = [dirs[idx[j]] for j in range((i+num_train_bin_file)*num_dirs_per_bin,  (i+1+num_train_bin_file)*num_dirs_per_bin)]
    #     val_dirs.append(tmp_dir)

    if not os.path.exists(tfrecords_data_path):
        os.mkdir(tfrecords_data_path)
    # print(num_train_bin_file)
    # print(num_val_bin_file)

    for i in range(num_bin_file):
        imageTensorMerged ,labelTensorMerged  = _get_one_tensor_object(train_dirs[i]) 
        print("i:",i)
        
        print("check:")
        print(i,imageTensorMerged.shape)
        print(i,labelTensorMerged.shape)
        

    # num_cores = multiprocessing.cpu_count()/2
    # Parallel(n_jobs=num_cores)(
    #     delayed(_get_one_tensor_object)(train_dirs[i]) for i in range(num_train_bin_file))

    # Parallel(n_jobs=num_cores)(
    #     delayed(_get_one_tensor_object)(val_dirs[i]) for i in range(num_val_bin_file))

    
    print("length_dir:",length_dirs)
    # print("num_train_bin_file):",num_train_bin_file)

    # print("num_val_bin_file):",num_val_bin_file)
    return imageTensorMerged,labelTensorMerged


def main():
    
    #processFile()
    imageTensorMerged,labelTensorMerged = processTensor()
    print("done")


if __name__ == "__main__":
    main()