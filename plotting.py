from datetime import datetime
from Tracking.iou_tracker import Tracker
from collections import OrderedDict
from collections import Counter

import numpy as np
import tensorflow as tf
import sqlite3
import cv2
import sys


sys.path.insert(2, '/Users/pascal/Coding/MP_bees/local_tensorflow/models')
sys.path.insert(3, '/Users/pascal/Coding/MP_bees/local_tensorflow/models/research')
sys.path.insert(4, '/Users/pascal/Coding/MP_bees/local_tensorflow/models/research/object_detection')
sys.path.insert(5, '/Users/pascal/Coding/MP_bees/object_tracking')

from object_detection.utils import label_map_util

DB_PATH = '/Users/pascal/Coding/MP_bees/object_tracking/bees.db'
# PATH_TO_VIDEO = '/Users/pascal/Coding/MP_bees/simple_object_tracking/videos/Froh_23_20191013_075648_540_M.mp4'
# PATH_TO_VIDEO = '/Users/pascal/Coding/MP_bees/videos/bees_2.mp4'
PATH_TO_VIDEO = '/Users/pascal/Coding/MP_bees/object_tracking/videos/' \
                '118_Doettingen_Hive1_200820_gopro8_1080_200fps_W_short.mp4'
# ALT_PATH_TO_VIDEO = '/content/gdrive/My Drive/Bees/data/Froh_23_20191013_075648_540_M.mp4'
ALT_PATH_TO_VIDEO = '/content/gdrive/My Drive/Bees/data/high_fps/' \
                    '118_Doettingen_Hive1_200820_gopro8_1080_100fps_W_short.mp4'
RUN_ID = 28
PATH_TO_FROZEN_GRAPH = '/Users/pascal/Coding/MP_bees/training_06_04/saved_model/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = '/Users/pascal/Coding/MP_bees/object_tracking/data/label_map.pbtxt'

vidcap = cv2.VideoCapture(PATH_TO_VIDEO)
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    break


img = cv2.imread('/Users/pascal/Coding/MP_bees/object_tracking/frame0.jpg')

width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

blank_image = np.zeros((height, width, 3), np.uint8)
blank_image[:, :] = (255, 255, 255)

img_center_x = width // 2 - 55
img_center_y = height // 2 - 20
# for united queens circle!
# cv2.circle(blank_image, (img_center_x, img_center_y), 135, (0, 0, 0), 5)
cv2.rectangle(blank_image, (660, 190), (1085, 260), (0, 0, 0),
              5)  # first tuple is the start, second tuple the end coordinates

gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 200, 800, 1)
contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

for cnt in contours:
    cv2.drawContours(blank_image, [cnt], -1, (36, 255, 12), 2)


plt.imshow(blank_image)
plt.show()


















# Basic Configuration
fig, axes = plt.subplots(ncols=2, figsize=(12, 12))
ax1, ax2 = axes
corr_matrix1 = D
corr_matrix2 = D[D_o]
columns1 = x_axis
columns2 = x_axis

# Heat maps.
im1 = ax1.matshow(corr_matrix1, cmap=plt.cm.get_cmap('coolwarm').reversed())
im2 = ax2.matshow(corr_matrix2, cmap=plt.cm.get_cmap('coolwarm').reversed())

# Formatting for heat map 1.
ax1.set_xticks(range(D.shape[1]))
ax1.set_yticks(range(D.shape[0]))
ax1.set_xticklabels(x_axis)
ax1.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
plt.setp(ax1.get_xticklabels())
plt.colorbar(im1, fraction=0.045, pad=0.05, ax=ax1)

# Formatting for heat map 2.
ax2.set_xticks(range(D[D_o].shape[1]))
ax2.set_yticks(range(D[D_o].shape[0]))
ax2.set_xticklabels(x_axis)
ax2.set_yticklabels(y_axis)
plt.setp(ax2.get_xticklabels())
plt.colorbar(im2, fraction=0.045, pad=0.05, ax=ax2,)

fig.tight_layout()
plt.show()


import matplotlib.pyplot as plt

matches = []
for i in total_matches:
    for j in i:
        matches.append(j)

no_matches = []
for i in total_no_matches:
    for j in i:
        no_matches.append(j)

all = matches + no_matches

colors = []
for i in all:
    if i in matches:
        colors.append('g')
    if i in no_matches:
        colors.append('r')

x, y = zip(*all)

for xa, ya, c in zip(x, y, colors):
    plt.scatter(xa, ya, color=c)

plt.xticks(np.arange(min(x),max(x), max(x)//10))
plt.yticks(np.arange(0.0,1.04, 0.05))

plt.show()


def get_iou_score(box1: np.ndarray, box2: np.ndarray):
    """
    calculate intersection over union cover percent
    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :return: IoU ratio if intersect, else 0
    """
    # first unify all boxes to shape (N,4)
    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU
    if union_area.all():
        intersect_ratio = intersect_area / union_area
    else:
        intersect_ratio = 0

    return base_mat * intersect_ratio


from scipy.spatial import distance as dist
dist.cdist(np.array(objectCoordinates), np.array(objectCoordinates),'euclidean')

zweihundert = [[[800, 473, 840, 558]],[[803, 458, 843, 529]]]
einhundert = [[[800, 473, 840, 558]],[[807, 441, 847, 512]]]
füfzg = [[[800, 473, 840, 558]],[[808, 410, 853, 475]]]
füfezwenzg = [[[800, 473, 840, 558]],[[814, 345, 854, 400]]]

