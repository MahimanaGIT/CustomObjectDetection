import rospy
import message_filters
import tensorflow.compat.v1 as tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from cv2 import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

print(tf.__version__)

PATH_TO_FROZEN_GRAPH = './models/fine_tuned_model/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = './data/label_map.pbtxt'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
NUM_CLASSES = 5
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
sess = tf.Session(graph=detection_graph, config = config)

cap = cv2.VideoCapture(2)

while True:
    # Read frame from camera
    ret, image_np = cap.read()
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Extract image tensor
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Extract detection boxes
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Extract detection scores
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # Extract detection classes
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # Extract number of detections
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        )
# Display output
    cv2.imshow('Person Detection', cv2.resize(image_np, (1200, 800)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

 # box = boxes.copy()

        # disparity=self.disparity_extractor.compute(image_np, right_disp)
        # disp = np.divide(disparity, 16.0)        
        # img_disp=cv2.normalize(disp,None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)

        # print("\n\nNew Frame")
        # for i in range (0, num_detections):
        #     if(scores[0,i] > 0.5):
        #         box[0,i,0] = boxes[0,i,0] * self.height
        #         box[0,i,1] = boxes[0,i,1] * self.width
        #         box[0,i,2] = boxes[0,i,2] * self.height
        #         box[0,i,3] = boxes[0,i,3] * self.width
        #         centerX = int(box[0,i,1] + (box[0,i,3] - box[0,i,1])/2)
        #         centerY = int(box[0,i,0] + (box[0,i,2] - box[0,i,0])/2)
        #         # print("Top Left Corner Coordinates: ({}, {})".format(box[0,i,1],box[0,i,0]))
        #         print("Center Coordinates: ({}, {})".format(centerX, centerY))
                
        #         center_disparity = disp[centerY, centerX]
        #         depth = 0.41245282 * self.fx / center_disparity
        #         x = (centerX - self.cx) * depth / self.fx
        #         y = (centerY - self.cy) * depth / self.fy
        #         z = depth

        #         print("3D Center Location: x = {}, y = {}, z = {}".format(x, y, z))
        #         if(classes[0,i] == 2):
        #             print('Processing Plant Detected with confidence {} % and box: {}'.format(scores[0,i] * 100, box[0,i,:]))
        #         elif (classes[0,i] == 3):
        #             print('Rock Detected with confidence {} and box: {}'.format(scores[0,i], box[0,i,:]))
        #         elif (classes[0,i] == 1):
        #             print('Cubesat Detected with confidence {} and box: {}'.format(scores[0,i], box[0,i,:]))
            
        
        # # Display output
        # try:
        #     cv2.imshow('CubeSat Detection', img_disp)
        # except:
        #     pass
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()