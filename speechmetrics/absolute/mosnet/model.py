import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
import librosa
import scipy
import numpy as np
import os
from ... import Metric
import hickle
import cloudpickle
import dill

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import keras2onnx

# prevent TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#import tensorrt as trt
#from tensorrt.parsers import uffparser
#from tensorrt import UffParser
#import uff
import deepdish
import snoop

class MOSNet(Metric):
    @snoop
    def __init__(self, window, hop=None):
        super(MOSNet, self).__init__(name='MOSNet', window=window, hop=hop)

        # constants
        self.fixed_rate = 16000
        self.mono = True
        self.absolute = True

        self.FFT_SIZE = 512
        self.SGRAM_DIM = self.FFT_SIZE // 2 + 1
        self.HOP_LENGTH = 256
        self.WIN_LENGTH = 512

        pre_trained_dir = os.path.dirname(__file__)
        try:
           self.model = tf.keras.models.load_model(os.path.join(pre_trained_dir, "keras-model"))
           #self.model = tf.keras.models.load_model("keras-model")
           print("got cached")
           return
        except:
           print("no cache")
           pass

        _input = keras.Input(shape=(None, 257))

        re_input = layers.Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)

        # CNN
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(re_input)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(conv1)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 3), activation='relu',
                 padding='same'))(conv1)

        conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(conv1)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(conv2)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 3), activation='relu',
                 padding='same'))(conv2)

        conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(conv2)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(conv3)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 3), activation='relu',
                 padding='same'))(conv3)

        conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(conv3)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                 padding='same'))(conv4)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 3), activation='relu',
                 padding='same'))(conv4)

        re_shape = layers.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)

        # BLSTM
        blstm1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3,
                 recurrent_dropout=0.3,
                 recurrent_constraint=max_norm(0.00001)),
            merge_mode='concat')(re_shape)

        # DNN
        flatten = TimeDistributed(layers.Flatten())(blstm1)
        dense1 = TimeDistributed(Dense(128, activation='relu'))(flatten)
        dense1 = Dropout(0.3)(dense1)

        frame_score = TimeDistributed(Dense(1), name='frame')(dense1)
        import warnings

        average_score = layers.GlobalAveragePooling1D(name='avg')(frame_score)


        self.model = Model(outputs=[average_score, frame_score], inputs=_input)
        # weights are in the directory of this file

        # load pre-trained weights. CNN_BLSTM is reported as best
        self.model.load_weights(os.path.join(pre_trained_dir, 'cnn_blstm.h5'))

        #temp_model_file = 'model.onnx'
        #keras2onnx.convert_keras(self.model, "mosnet")
        #tf.saved_model.save(self.model, "./models/")
        #dill.dump(self.model, "test.cloudpickle")
        #uff_model = uff.from_tensorflow(self.model, ['avg'])
        #deepdish.io.save('test.h5', self.model)
        #converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]

        #tflite_model = converter.convert()
        #tf.saved_model.save(self.model, "./models")
        self.model.compile()
        #tf.keras.models.save_model(self.model, os.path.join(pre_trained_dir, "keras-model"))
        #print("saved")




    def test_window(self, audios, rate):
        # stft. D: (1+n_fft//2, T)
        linear = librosa.stft(y=np.asfortranarray(audios[0]),
                              n_fft=self.FFT_SIZE,
                              hop_length=self.HOP_LENGTH,
                              win_length=self.WIN_LENGTH,
                              window=scipy.signal.hamming,
                              )

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft/2, T)

        # shape in (T, 1+n_fft/2)
        mag = np.transpose(mag.astype(np.float32))

        # now call the actual MOSnet
        return {'mosnet':
                self.model.predict(mag[None, ...], verbose=0, batch_size=1)[0]}
