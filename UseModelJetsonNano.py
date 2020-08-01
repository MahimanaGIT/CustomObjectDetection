import tensorflow as tf

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

# The TensorRT inference graph file downloaded from Colab or your local machine.
pb_fname = "./models/trt_model/trt_graph.pb"

print("reading graph")
trt_graph = get_frozen_graph(pb_fname)
print("Done reading graph")
                                                                                                                                
input_names = ['image_tensor']
# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')                                                                                                                                                                                                                               
tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

import cv2
cap = cv2.VideoCapture(0)

while True:
    return_key, image = cap.read()
    image = cv2.resize(image, (300, 300))
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
    tf_input: image[None, ...]
    })

    boxes = boxes[0]  # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])
#     print(boxes[0])
#     print(scores[0])
#     for i in range (0, num_detections):
    if(scores[0] > 0.5):
        b1 = int(boxes[0][0] * 299)
        b2 = int(boxes[0][1] * 299)
        b3 = int(boxes[0][2] * 299)
        b4 = int(boxes[0][3] * 299)
        image[b1:b3, b2] = [255, 255, 0]
        image[b3, b2:b4] = [255, 255, 0]
        image[b1:b3, b4] = [255, 255, 0]
        image[b1, b2:b4] = [255, 255, 0]
    cv2.imshow('Person Detection', cv2.resize(image, (1200, 800)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    