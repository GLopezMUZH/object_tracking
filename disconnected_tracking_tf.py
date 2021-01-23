from datetime import datetime
from pyimagesearch.iou_tracker import Tracker
from pyimagesearch.trackableobject import TrackableObject

from imutils.video import FPS
from imutils.video import FPS
from imutils.video import VideoStream
import numpy as np
import tensorflow as tf
import sqlite3
import dlib
import math
import cv2
import sys

sys.path.insert(2, '/Users/pascal/Coding/MP_bees/local_tensorflow/models')
sys.path.insert(3, '/Users/pascal/Coding/MP_bees/local_tensorflow/models/research')
sys.path.insert(4, '/Users/pascal/Coding/MP_bees/local_tensorflow/models/research/object_detection')
sys.path.insert(5, '/Users/pascal/Coding/MP_bees/simple_object_tracking')

from utils import label_map_util
from utils import visualization_utils as vis_util

DB_PATH = '/Users/pascal/Coding/MP_bees/simple_object_tracking/bees.db'
# PATH_TO_VIDEO = '/Users/pascal/Coding/MP_bees/simple_object_tracking/videos/Froh_23_20191013_075648_540_M.mp4'
PATH_TO_VIDEO = '/Users/pascal/Coding/MP_bees/videos/bees_2.mp4'

# ALT_PATH_TO_VIDEO = '/content/gdrive/My Drive/Bees/data/Froh_23_20191013_075648_540_M.mp4'
ALT_PATH_TO_VIDEO = '/content/gdrive/My Drive/Bees/data/bees_2.mp4'
RUN_ID = 18
PATH_TO_FROZEN_GRAPH = '/Users/pascal/Coding/MP_bees/training_06_04/saved_model/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = '/Users/pascal/Coding/MP_bees/simple_object_tracking/data/label_map.pbtxt'
NUM_CLASSES = 1
THRESHOLD = 0.1
MASK = False
BBOXES = False
DISTANCE = False
SHOW = True
IOU_MATCHING_H = False

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()


def get_coordinates_from_db(run_id, video, frame):
    c.execute(
        "select * from coordinates where run_id = {} and video = '{}' and frame = {}".format(run_id, video, frame))
    return c.fetchall()



ct = Tracker(50,2,5,0.5)
trackableObjects = {}
trackers = []
cap = cv2.VideoCapture(PATH_TO_VIDEO)
fps = FPS().start()

totalFrames = 0
totalIn = 0
totalOut = 0

image_center_x = 420
image_center_y = 350

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

dateTimeObj = datetime.now()
time_stamp = dateTimeObj.strftime("%d_%b_%Y_%H_%M_%S.%f")

outname = 'bee_output_{}.avi'.format(time_stamp)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(outname, fourcc, 10.0, (int(width), int(height)))
detections = []

# Create blank image for entrance contour detection

blank_image = np.zeros((height, width, 3), np.uint8)
blank_image[:, :] = (255, 255, 255)

img_center_x = width // 2 - 55
img_center_y = height // 2 -20
cv2.circle(blank_image, (img_center_x, img_center_y), 135, (0, 0, 0), 5)
# cv2.rectangle(blank_image, (width//6, height//3), ((width//6)*5, (height//8)*4), (0,0,0), 5)
gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 200, 800, 1)
contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (127, 127, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]

# Detection

frame = 0
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            frame += 1
            w_h = tf.constant([width, height, width, height], dtype=tf.float32)
            ret, image_np = cap.read()
            if ret:
                rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                image_np_expanded = np.expand_dims(image_np, axis=0)
                # cv2.circle(image_np, (img_center_x, img_center_y), 140, (0, 0, 0), 5)

                if frame > 1:
                    coordinates_n_minus_1 = get_coordinates_from_db(RUN_ID, ALT_PATH_TO_VIDEO, frame-1)
                coordinates = get_coordinates_from_db(RUN_ID, ALT_PATH_TO_VIDEO, frame)

                height, width, channels = image_np.shape

                rects = []
                boxes = []
                scores = []
                classes = []
                bee_counter = 0
                for i in range(len(coordinates)):
                    r_id, f_name, fr, b_id, xmin, xmax, ymin, ymax, X, Y, conf = coordinates[i]
                    if conf >= THRESHOLD:
                        for cnt in contours:
                            cv2.drawContours(image_np, [cnt], -1, (36, 255, 12), 2)
                            res = cv2.pointPolygonTest(cnt, (X,Y), False)
                            if res == -1 or res == 1 or res == 0:
                                bee_counter += 1
                                rects.append([xmin, ymin, xmax, ymax])
                                boxes.append([ymin / height, xmin / width,ymax / height, xmax / width])
                                scores.append(conf)
                                classes.append(1.0)

                                center = (X, Y)
                                # construct a dlib rectangle object from the bounding
                                # box coordinates and then start the dlib correlation
                                # tracker
                                dlib_tracker = dlib.correlation_tracker()
                                rect = dlib.rectangle(xmin, ymin, xmax, ymax)
                                dlib_tracker.start_track(rgb, rect)
                                #TODO:
                                # add the tracker to our list of trackers so we can
                                # utilize it during skip frames
                                trackers.append(dlib_tracker)
                boxes = np.array([boxes])
                scores = np.array([scores])
                classes = np.array(classes)
                num_detections = len(coordinates)
                objects = ct.update(rects)

                for j in range(len(ct.tracks)):
                    if (len(ct.tracks[j].trace) > 1):
                        x = int(ct.tracks[j].trace[-1][0, 0])
                        y = int(ct.tracks[j].trace[-1][0, 1])
                        cv2.putText(image_np, str(ct.tracks[j].trackId), (x - 10, y - 20), 0, 0.5, (0,255,0), 2)
                        for k in range(len(ct.tracks[j].trace)):
                            x = int(ct.tracks[j].trace[k][0, 0])
                            y = int(ct.tracks[j].trace[k][0, 1])
                            cv2.circle(image_np, (x, y), 3, (0,255,0), -1)

                for (objectID, coordinates) in objects.items():
                    centroid_x = coordinates[0] + (coordinates[2]-coordinates[0])//2
                    centroid_y = coordinates[1] + (coordinates[3]-coordinates[1])//2
                    centroid = (centroid_x,centroid_y)


                    for cnt in contours:
                        cv2.drawContours(image_np, [cnt], -1, (36, 255, 12), 2)
                        res = cv2.pointPolygonTest(cnt, tuple(centroid), False)
                        if res == -1 or res == 1 or res == 0:
                            # draw both the ID of the object and the centroid of the
                            # object on the output frame
                            # text = "ID {}".format(i)
                            # cv2.putText(image_np, text, (x - 10, y - 10),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # cv2.circle(image_np, (x, y), 4, (0, 255, 0), -1)
                            to = trackableObjects.get(objectID, None)
                            text = "ID {}".format(objectID)
                            # cv2.circle(image_np, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                            # if there is no existing trackable object, create one
                            if to is None:
                                to = TrackableObject(objectID, centroid)
                                # otherwise, there is a trackable object so we can utilize it
                                # to determine direction
                            else:
                                # the difference between the y-coordinate of the *current*
                                # centroid and the mean of *previous* centroids will tell
                                # us in which direction the object is moving (negative for
                                # 'up' and positive for 'down')
                                y = [c[1] for c in to.centroids]
                                x = [c[0] for c in to.centroids]

                                direction_y = centroid[1] - np.mean(y)
                                direction_x = centroid[0] - np.mean(x)

                                if DISTANCE:
                                    to.centroids.append(centroid)
                                    distance_to_center = math.hypot(image_center_x - x[-1], image_center_y - y[-1])
                                    # text = "dist {}".format(int(distance_to_center))
                                    cv2.arrowedLine(image_np, (int(np.mean(x)), int(np.mean(y))),
                                                    (centroid[0], centroid[1]), (0, 255, 0), 5)
                                # check to see if the object has been counted or not
                                if not to.counted:
                                    # if the direction is negative (indicating the object
                                    # is moving up) AND the centroid is above the center
                                    # line, count the object
                                    #if direction_x < 0 and centroid[1] < height // 2:
                                    if res == 1:
                                        totalIn += 1
                                        to.counted = True
                                        # if the direction is positive (indicating the object
                                        # is moving down) AND the centroid is below the
                                        # center line, count the object
                                    #elif direction_x > 0 and centroid[1] > height // 2:
                                    if res == -1:
                                        totalOut += 1
                                        to.counted = True

                            # store the trackable object in our dictionary
                            trackableObjects[objectID] = to

                            # cv2.putText(image_np, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                print("num detections: {}".format(int(len(coordinates))))
                # Visualization of the results of a detection.
                if BBOXES:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        min_score_thresh=THRESHOLD,
                        line_thickness=1)
                # construct a tuple of information we will be displaying on the
                # frame
                info = [("Nr of Bees", int(bee_counter)), ("out", totalOut), ("in", totalIn)]
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(image_np, text, (10, int(height) - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0),
                                1)
                    # cv2.putText(image_np, "Nr of Bees: {}".format(int(len(coordinates))), (10,20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if ret:
                    out.write(cv2.resize(image_np, (int(width), int(height))))

                # Display output
                if SHOW:
                    cv2.imshow('', cv2.resize(image_np, (int(width), int(height))))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

cap.release()
out.release()
cv2.destroyAllWindows
conn.close()
# stop the timer and display FPS information
fps.stop()
