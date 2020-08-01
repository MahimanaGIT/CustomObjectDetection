# Custom Object Detection using Tensorflow On Jetson Nano

This tutorial is an implementation of full pipeline from creating a custom dataset, annotate it, training SSD-Mobilenet model using transfer learning on a custom dataset and deploy it NVIDIA's Jetson Nano

Inspired by following tutorials:
> [Step by Step: Build Your Custom Real-Time Object Detector](https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d)


> [How to run TensorFlow Object Detection model on Jetson Nano](https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/)

<img src="https://github.com/MahimanaGIT/PersonTracking/blob/master/images/gazebo_detection.gif" />

### Requirements:
1. Tensorflow 1.15
2. Python 2.7

The following steps are for making a custom dataset, if anyone wants to save image from a camera, the following repository will be useful:
[Dataset Builder](https://github.com/MahimanaGIT/DatasetBuilder)

Clone the repository for object_detection_api in any location you feel comfortable: [object_detection_api](https://github.com/tensorflow/models) from tensorflow

Annotate the images of the dataset using [LabelImg](https://github.com/tzutalin/labelImg), save all the labels in "annotations" folder in the repository.


The repository should be of the following for:
> Object Detection Repostiory:
>   - data
>        - images
>            - ".jpg" files
>       - annotations
>            - ".xml" files from LabelImg
>        - test_labels
>            - Separate the  test labels (".xml" files)
>        - train_labels
>            - Separate the train labels (".xml" files)
>   - models
>        - pretrained_model
>        - fine_tune_model
>        - training
>        - trt_model
>   - All the codes in the repository    

## Train model for custom object detection:

Step 1: Saving images for the dataset in "data/images" folder

Step 2: Saving corresponding xml files for every images of the dataset using LabelImg in "data/annotations" folder

Step 3: Separate corresponding xml files in "test_labels" and "train_labels" folder in "data" folder

Step 4: Converting corresponding xml files for "test_labels" and "training_labels" to CSV files and label_map.pbtxt file using the script:

>   python Preprocessing.py

This script will convert the ".xml" files from the two corresponding folders and save them in collection in ".csv" files, as well as generate a "label_map.pbtxt" file which contains all the classes name with corresponding id number in the "data" folder.

Step 5: Exporting the python path to add the object detection from tensorflow object detection API: 

    "export PYTHONPATH=$PYTHONPATH:~/repo/object_detection/object_detection/models/research/:~/repo/object_detection/object_detection/models/research/slim/"

Step 6: Run the following command in terminal from the directory "object_detection/models/research/" for compiling the proto buffers:

>   protoc object_detection/protos/*.proto --python_out=."

Step 7: Run the script to check if everything is OK in the "research" folder of the [object_detection_api](https://github.com/tensorflow/models) from tensorflow

>   python object_detection/builders/model_builder_test.py

Step 8: Generate TF Records which saves the data as binary strings, surely change the classes information in the script according to the number of classes and name of each class in the script and then run:

>   python GeneratingTFRecords.py

Step 9: Select and download the model using the script which will save the pretrained model in the folder "./models/pretrained_model" and extract it:

>   python3 SelectingAndDownloadingModel.py

python2 was causing library problem in downloading the dataset

Step 10: Configure Model Training Pipeline i.e. "pipeline.config" file in the folder "./models/pretrained_model" from the existing sample of config file ***DO NOT USE THE FILE IN PRETRAINED_MODEL***, use the file present in object_detection api folder "/object_detection/models/research/object_detection/samples/config/ssd_mobilenet_v2_coco.config":

1. Change the number of classes that you are detecting
2. Change the path for "fine_tubne_checkpoint", set the path to"./models/pretrained_model/model.ckpt", this file will ***NOT*** be present actually.
3. Change the path to train input_path to "./data/train_labels.record", evaluation input_path to "./data/test_labels.record"
4. Change the label_map_path in both train and evaluation input reader to the same label_map.pbtxt file "./data/label_map.pbtxt"

Step 11: To launch tensorboard, execute the following command in a new terminal:

>   tensorboard --logdir=./models/training

Step 13: Start the training running the script and giving the arguments:

>   python model_main.py --pipeline_config_path=./models/pretrained_model/pipeline.config --model_dir=./models/training

Step 14: Change the classes and all the class id number in the corresponding files.

Step 15: Export the best model using the script

>   python ExportTrainedModel.py

## For Deploying on PC:

Use the frozen path from "./models/fine_tuned_model/frozen_inference_graph.pb" with the file name "frozen_inference_graph.pb" and "./data/label_map.pbtxt" and run the script:

>   python UseModel.py

## For Deploying on Jetson Nano


Use the existing trained model and optimize it to run on Jetson nano by making some changes to the existing graph:

1. The score threshold is set to 0.3, so the model will remove any prediction results with confidence score lower than the threshold.
2. IoU(intersection over union) threshold is set to 0.5 so that any detected objects with same classes overlapped will be removed. 
3. Fix the batch size 1 for memory constaints on Jetson Nano

Run the script to make the changes to frozen graph and save the"trt_graph" in the folder "./models/trt_model":

>   python ConvertToTRTModel.py

For running the model on Jetson Nano, use the following script to run on webcam: 

>   python UseModelJetsonNano.py

    or
    
Using the jupyter notebook:

>   UseModelJetsonNanoNotebook.ipynb


<img src="https://github.com/MahimanaGIT/PersonTracking/blob/master/images/person_detection.gif" width="500" height="500" />

Note on using on Jetson Nano:

1. The graph takes appoximately 2-3 minutes to read the graph and 1-2 minutes for importing the graph
2. Converting the model to pure tensorrt model will be covered in next tutorial.

Debugging:

Using TFRecord viewer for verifying the .record file, execute:

>    python3 tfviewer.py /home/mahimana/Documents/Deep_Learning/ObjectDetection/data/train_labels.record --labels-to-highlight='cubesat, rock, processing_plant'

