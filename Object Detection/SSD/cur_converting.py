import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pickle
import cv2
import numpy as np

from model.SSD_model import SSD
from loss.MultiBoxLoss import MultiboxLoss

NUM_CLASSES = 21
BATCH_SIZE = 8
IMAGE_SIZE = 224

# path
crack_pkl = './VOC/VOC2007.pkl'
IMAGE_PATH = './VOC/images/'

# .pkl에서 데이터를 불러옵니다.
gt = pickle.load(open(crack_pkl, 'rb'))

# gt의 key는 이미지 이름으로 이루어져 있습니다.
# gt의 value_list는 이미지에 존재하는 객체 수로 이루어져 있다.
# gt의 value는 총 24 길이로 이루어져 있는데,
# 앞의 첫 4개 인덱스는 xmin, xmax, ymin, ymax 좌표입니다.
# 나머지 20개는 클래스를 나타냅니다.
keys = sorted(gt.keys())

# 학습 및 검증 데이터를 8:2로 나누도록 하겠습니다.
num_train = int(round(0.8 * len(keys)))

# 3962
train_keys = keys[:num_train]
# 990
val_keys = keys[num_train:]

num_val = len(val_keys)


# 데이터셋 객체는 이미지(로드)와 레이블을 반환하도록 구성합니다.
# 먼저, 이미지 경로와 해당 값을 리스트에 저장해두도록 하겠습니다.
image_dir_list = list()
value_list = list()

for i in range(BATCH_SIZE):
    image_dir_list.append(train_keys[i])
    value_list.append(gt[train_keys[i]])

image_dir_list = np.array(image_dir_list)

value_list = np.array(value_list)

# ragging data.
ragged_value_list = tf.ragged.constant(value_list)

image_dir_ds = tf.data.Dataset.from_tensor_slices(image_dir_list)
value_ds = tf.data.Dataset.from_tensor_slices(ragged_value_list)
# use 8 batch size
value_ds = value_ds.batch(8)

def get_imageLabel(image_dir):
    image = tf.io.read_file(IMAGE_PATH + image_dir)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])

    return image

image_ds = image_dir_ds.map(get_imageLabel)
image_ds = image_ds.batch(8)

image = None
value = None

for i in image_ds:
    image = i
for j in value_ds:
    value = j

# make model
input_shape = (224, 224, 3)
model = SSD(input_shape, num_classes = NUM_CLASSES)

optimizer = Adam()
train_loss = MultiboxLoss(BATCH_SIZE)

with tf.GradientTape() as tape:
    # predictions shape: (None, 938, 29)
    # value shape: (Object Number, None)
    predictions = model(image)
    loss = train_loss.comute_loss(value, predictions)