import os
import cv2
import lmdb
import numpy as np
import argparse
import shutil
import sys


def checkValid(imgBin):
    if imgBin is None:
        return False

    try:
        imgBuffer = np.fromstring(imgBin, dtype=np.uint8)
        img = cv2.imdecode(imgBuffer, cv2.IMREAD_GRAYSCALE)
        img_height, img_width = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if img_height * img_width == 0:
            return False

    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, alphabetList=None, checkValid=True):
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    assert (len(imagePathList) == len(labelList))
    num_samples = len(imagePathList)
    db = lmdb.open(outputPath, map_size=2511627776)
    cache = {}
    count = 1
    for i in range(num_samples):
        imagePath = ''.join(imagePathList[i]).split()[0].replace('\n', '').replace('\r\n', '')
        # print(imagePath)
        label = ''.join(labelList[i])
        # print(label)

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % count
        labelKey = 'label-%09d' % count
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if alphabetList:
            alphabetKey = 'lexicon-%09d' % count
            cache[alphabetKey] = ' '.join(alphabetList[i])
        if count % 1000 == 0:
            writeCache(db, cache)
            cache = {}
            print('Written %d / %d' % (count, num_samples))
        count += 1
    num_samples = count - 1
    cache['num-samples'] = str(num_samples)
    writeCache(db, cache)
    db.close()
    print('Created dataset with %d samples' % num_samples)



def read_from_file(file_path):
    image_path_list = []
    label_list = []
    f = open(file_path, encoding='utf-8')
    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line1 or not line2:
            break
        line1 = line1.replace('\r', '').replace('\n', '')
        line2 = line2.replace('\r', '').replace('\n', '')
        image_path_list.append(line1.split()[0])
        image_path_list.append(line2.split()[0])
        label_list.append(line1.split()[1])
        label_list.append(line2.split()[1])
    return image_path_list, label_list


def read_from_folder(folder_path):
    image_path_list = []
    label_list = []
    pics = os.listdir(folder_path)
    print(pics)
    pics.sort(key=lambda i: len(i))
    for pic in pics:
        image_path_list.append(folder_path + '/' + pic)
        label_list.append(pic.split('_')[0])

    return image_path_list, label_list



def test(test_number, image_path_list, label_list):
    print('The first line should be the path to image and the second line should be corresponding image label')
    for i in range(test_number):
        print('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='path to folder containing the img',
                        default="./coco-text_dataset/val/")
    parser.add_argument('--file', type=str, help='path to file which contains the image path and label',
                        default='./coco-text_dataset/val.txt')
    args = parser.parse_args()

    if args.file is not None:
        image_path_list, label_list = read_from_file(args.file)
        createDataset('./lmdb_val', image_path_list,
                      label_list)
        test(5, image_path_list, label_list)
    elif args.folder is not None:
        image_path_list, label_list = read_from_folder(args.folder)
        createDataset('./lmdb_val', image_path_list,
                      label_list)
        test(5, image_path_list, label_list)
    else:
        sys.exit()
