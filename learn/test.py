
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
import random
import  PIL.Image
'''
def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      {"height": tf.io.FixedLenFeature([], dtype=tf.int64),
       "width": tf.io.FixedLenFeature([], dtype=tf.int64),
       "image_raw":  tf.io.FixedLenFeature([], dtype=tf.string),
       "label": tf.io.FixedLenFeature([], dtype=tf.int64)}
  )

dental_files = []
dental_files.append('tooth.tfrecord')

for batch in tf.data.TFRecordDataset([dental_files]).map(decode_fn):
    print(tf.keras.backend.get_value(batch['height']))
    print(tf.keras.backend.get_value(batch['width']))
    print(tf.keras.backend.get_value(batch['image_raw']))
    print(tf.keras.backend.get_value(batch['label']))
    print("================")
'''

counter = 0

import tensorflow as tf
tf.enable_eager_execution()
raw_dataset = tf.python_io.tf_record_iterator("tooth.tfrecord")
label = -1
for str_rec in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(str_rec)
    for key, feature in example.features.feature.items():
        if key == 'image/object/class/label':
            label = list(feature.int64_list.value)
        if key == 'image/encoded':
            if label != -1:
                image = feature.bytes_list.value[0]
                label = max(label,key = label.count)
                if label == 6:
                    f = open('Dataset/Normal/'+str(counter)+".png", 'wb')
                    f.write(image)
                    f.close()
                    counter = counter + 1
                if label != 6 and label != -1:
                    f = open('Dataset/Disease/'+str(counter)+".png", 'wb')
                    f.write(image)
                    f.close()
                    counter = counter + 1
                print(str(label))
                
                label = -1
            













            
      
