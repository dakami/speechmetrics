import lazy_import
#lazy_import.lazy_module("numpy")
#lazy_import.lazy_module("librosa")
#lazy_import.lazy_module("scipy.io.wavfile")
import io
import snoop
import librosa
import numpy as np
import os
import scipy.io.wavfile
#from speechmetrics import Metric

import tflite_runtime.interpreter as tflite

# prevent TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MyMOS():
    @snoop
    def __init__(self, window=None, verbose=0, hop=None):
        #super(MyMOS, self).__init__(name='MOSNet', window=window, hop=hop)

        # constants
        self.fixed_rate = 16000
        self.mono = True
        self.absolute = True
        self.verbose=verbose
        self.window=window

        self.FFT_SIZE = 512
        self.SGRAM_DIM = self.FFT_SIZE // 2 + 1
        self.HOP_LENGTH = 256
        self.WIN_LENGTH = 512
        self.model = dotdict({'predict': None})
        pre_trained_dir = os.path.dirname(__file__)

        try:
            tflite_model=open(os.path.join(pre_trained_dir, 'mosnet.tflite'), "rb").read()
            self.interpreter = tflite.Interpreter(model_content=tflite_model)

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            @snoop
            def predict(mag,verbose,batch_size):
              self.interpreter.resize_tensor_input(self.input_details[0]['index'], mag.shape)
              self.interpreter.allocate_tensors()
              self.interpreter.set_tensor(self.input_details[0]['index'], mag)
              self.interpreter.invoke()
              return self.interpreter.get_tensor(self.output_details[0]['index'])
            self.model.predict = predict
            return
        except: raise

    @snoop
    def test(self, *test_files, array_rate=None):
        """loading sound files and making sure they all have the same lengths
        (zero-padding to the largest). Also works with numpy arrays.
        Then, calling the `test_window` function that should be specialised
        depending on the metric."""

        # imports
        import soundfile as sf
        import resampy
        from museval.metrics import Framing
        import numpy as np

        audios = []
        maxlen = 0
        if isinstance(test_files, str):
            test_files = [test_files]
        if self.absolute and len(test_files) > 1:
            if self.verbose:
                print('  [%s] is absolute. Processing first file only'
                      % self.name)
            test_files = [test_files[0],]

        for file in test_files:
            # Loading sound file
            if isinstance(file, str):
                audio, rate = sf.read(file, always_2d=True)
            else:
                rate = array_rate
                if rate is None:
                    raise ValueError('Sampling rate needs to be specified '
                                     'when feeding numpy arrays.')
                audio = file
                # Standardize shapes
                if len(audio.shape) == 1:
                    audio = audio[:, None]
                if len(audio.shape) != 2:
                    raise ValueError('Please provide 1D or 2D array, received '
                                     '{}D array'.format(len(audio.shape)))

            if self.fixed_rate is not None and rate != self.fixed_rate:
                if self.verbose:
                    print('  [%s] preferred is %dkHz rate. resampling'
                          % (self.name, self.fixed_rate))
                audio = resampy.resample(audio, rate, self.fixed_rate, axis=0)
                rate = self.fixed_rate
            if self.mono and audio.shape[1] > 1:
                if self.verbose:
                    print('  [%s] only supports mono. Will use first channel'
                          % self.name)
                audio = audio[..., 0, None]
            if self.mono:
                audio = audio[..., 0]
            maxlen = max(maxlen, audio.shape[0])
            audios += [audio]

        for index, audio in enumerate(audios):
            if audio.shape[0] != maxlen:
                new = np.zeros((maxlen,) + audio.shape[1:])
                new[:audio.shape[0]] = audio
                audios[index] = new

        if self.window is not None:
            framer = Framing(self.window * rate,
                             self.hop * rate, maxlen)
            nwin = framer.nwin
            result = {}
            for (t, win) in enumerate(framer):
                result_t = self.test_window([audio[win] for audio in audios],
                                            rate)
                for metric in result_t.keys():
                    if metric not in result.keys():
                        result[metric] = np.empty(nwin)
                    result[metric][t] = result_t[metric]
        else:
            result = self.test_window(audios, rate)
        return result



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
                self.model.predict(mag.reshape(1,-1,257), verbose=0, batch_size=1)[0]}
  
    def read_wav(self,raw):
        (rate, wav) = scipy.io.wavfile.read(io.BytesIO(raw))
        fwav=wav.astype(float)
        fwav/=32768.
        return self.test(fwav, array_rate=rate)["mosnet"][0]


