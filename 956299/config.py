# config file for main.py
DATA_DIR = "./coco-text_dataset/" # path to the dataset
JSON_DIR = "./coco-text_dataset/cocotext.v2.json"  # path to the json
LMDB_TRAIN_ROOT = './lmdb_train' # path to the train dataset
LMDB_VAL_ROOT = './lmdb_val'  # path to the test dataset
NUM_WORKERS = 0  # anything more than zero throws pickle envrionment error
BATCH_SIZE = 100 # dataset batch size
IMG_HEIGHT = 32  # tensor img height
NGPU = 1  # number of gpus for multi gpu support
PRE_TRAINED = ''  # for loading pretrained model to resume training
MANUAL_SEED = 1234  # for ensuring we can repeat experiments
IMG_WIDTH = 100  # tensor img width
EPOCHS = 50  # number of iterations of the dataset
H_LSTM = 256  # number of lstm hidden layers
L_RATE = 0.001  # learning rate of the model
ALPHABET = "'0123456789abcdefghijklmnopqrstuvwxyz«"".&-ãš›é~^¬µ`“”ï°á €¢/ú©?()@:;,+=_$£!%> " '"'"<""く ᛫| /ⓔ[]{}*#'â '‚í''\xad' \ ®ºë""π"
IS_CUDA = True  # set this to false if you dont have a cuda enabled GPU
DISP_INTERVAL = 500  # Display Interval
SAVE_INTERVAL = 500  # save interval
VAL_INTERVAL = 500  # validation interval
NUM_TEST_DISP = 10  # num sample to be displayed when testing
KEEP_RATIO = True # hold image ratio when resizing
CTCNANLOSSERROR = True # evading CTC loss nan bug
TEST = None  # to check for folder where checkpoint is saved
