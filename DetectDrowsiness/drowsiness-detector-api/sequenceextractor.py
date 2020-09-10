import numpy as np
import os.path
import csv
import glob
import tensorflow as tf
import h5py as h5py
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import natsort
import sys

class SequenceExtractor:
    def __init__(self, seq_length=50, model_path='./output_graph.pb'):
        self.seq_length = seq_length
        self.model_path = model_path

    def featureExtractor(self, image_path):
        with open(self.model_path, 'rb') as graph_file:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(graph_file.read())
            tf.import_graph_def(graph_def, name='')

        with tf.compat.v1.Session() as sess:
            pooling_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            image_data = tf.compat.v1.gfile.FastGFile(image_path, 'rb').read()	
            pooling_features = sess.run(pooling_tensor, {'DecodeJpeg/contents:0': image_data})
            pooling_features = pooling_features[0]
        return pooling_features

    def extractFeaturesFromImagePath(self, image_path, sequence_path):
        allFrames = glob.glob(os.path.join(image_path + '/*jpg'))
        allFrames = natsort.natsorted(allFrames,reverse=False)
        sequence = []
        counter = 0
        for image in allFrames: 
            with tf.Graph().as_default():
                imageFeature = self.featureExtractor(image)
                counter+=1
                sequence.append(imageFeature)
            if counter % self.seq_length == 0: 
                np.save(sequence_path+str(counter)+'.npy', sequence)
                sequence = []


    def extractTraining(self, training_data_path, sequence_path):
        with open(training_data_path,'r') as f:
            reader = csv.reader(f)
        for videos in reader:
            path = os.path.join('data', 'sequences', videos[0], videos[2] + '-' + str(self.seq_length) + '-features')
            path_frames = os.path.join('data', videos[0], videos[1])
            filename = videos[2]
            image_path = path_frames + filename
            #print(filename)
            self.extractFeaturesFromImagePath(image_path, sequence_path)
            print('Sequences saved successfully')

    																																																																
