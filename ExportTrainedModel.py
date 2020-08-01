import os
import numpy as np
import re

#dir where the model will be saved
output_directory = './models/fine_tuned_model/'

lst = os.listdir('./models/training/')
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')

last_model_path = os.path.join('./models/training/', last_model)

print(last_model_path)

os.system("python ~/repo/object_detection/object_detection/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=./models/pretrained_model/pipeline_1.config \
    --output_directory={} \
    --trained_checkpoint_prefix={}".format(output_directory, last_model_path))