import numpy as np, os, functools, pathlib, onnxruntime as ort
from collections import deque
from typing import List, Union, Callable

class AudioFeatures():
    def __init__(self, melspec_model_path: str = "melspectrogram.onnx", embedding_model_path: str = "embedding_model.onnx", sr: int = 16000,
                 ncpu: int = 1, inference_framework: str = "onnx", device: str = 'gpu'):
        
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = ncpu
        sessionOptions.intra_op_num_threads = ncpu
        self.melspec_model = ort.InferenceSession(melspec_model_path, sess_options=sessionOptions, providers=["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"])
        self.onnx_execution_provider = self.melspec_model.get_providers()[0]
        self.melspec_model_predict = lambda x: self.melspec_model.run(None, {'input': x})
        self.embedding_model = ort.InferenceSession(embedding_model_path, sess_options=sessionOptions, providers=["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"])
        self.embedding_model_predict = lambda x: self.embedding_model.run(None, {'input_1': x})[0].squeeze()
        self.raw_data_buffer: Deque = deque(maxlen=sr*10)
        self.melspectrogram_buffer = np.ones((76, 32))  # n_frames x num_features
        self.melspectrogram_max_len = 10*97  # 97 is the number of frames in 1 second of 16hz audio
        self.accumulated_samples = 0  # the samples added to the buffer since the audio preprocessor was last called
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))
        self.feature_buffer_max_len = 120  # ~10 seconds of feature buffer history

    def _get_melspectrogram(self, x: Union[np.ndarray, List], melspec_transform: Callable = lambda x: x/10 + 2):
        x = np.array(x).astype(np.int16) if isinstance(x, list) else x
        x = x[None, ] if len(x.shape) < 2 else x
        x = x.astype(np.float32) if x.dtype != np.float32 else x
        outputs = self.melspec_model_predict(x)
        spec = np.squeeze(outputs[0])
        spec = melspec_transform(spec)
        return spec

    def _get_embeddings(self, x: np.ndarray, window_size: int = 76, step_size: int = 8, **kwargs):
        spec = self._get_melspectrogram(x, **kwargs)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size:  # truncate short windows
                windows.append(window)
        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        embedding = self.embedding_model_predict(batch)
        return embedding

    def _streaming_melspectrogram(self, n_samples):
        self.melspectrogram_buffer = np.vstack((self.melspectrogram_buffer, self._get_melspectrogram(list(self.raw_data_buffer)[-n_samples-160*3:])))
        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def _buffer_raw_data(self, x): self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

    def _streaming_features(self, x): # Add raw audio data to buffer, temporarily storing extra frames if not an even number of 80 ms chunks
        processed_samples = 0
        if self.raw_data_remainder.shape[0] != 0:
            x = np.concatenate((self.raw_data_remainder, x))
            self.raw_data_remainder = np.empty(0)
        if self.accumulated_samples + x.shape[0] >= 1280:
            remainder = (self.accumulated_samples + x.shape[0]) % 1280
            if remainder != 0:
                x_even_chunks = x[0:-remainder]
                self._buffer_raw_data(x_even_chunks)
                self.accumulated_samples += len(x_even_chunks)
                self.raw_data_remainder = x[-remainder:]
            elif remainder == 0:
                self._buffer_raw_data(x)
                self.accumulated_samples += x.shape[0]
                self.raw_data_remainder = np.empty(0)
        else:
            self.accumulated_samples += x.shape[0]
            self._buffer_raw_data(x)
        if self.accumulated_samples >= 1280 and self.accumulated_samples % 1280 == 0:# Only calculate melspectrogram once minimum samples are accumulated
            self._streaming_melspectrogram(self.accumulated_samples)
            for i in np.arange(self.accumulated_samples//1280-1, -1, -1):# Calculate new audio embeddings/features based on update melspectrograms
                ndx = -8*i
                ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
                x = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
                if x.shape[1] == 76:
                    self.feature_buffer = np.vstack((self.feature_buffer,
                                                    self.embedding_model_predict(x)))
            processed_samples = self.accumulated_samples# Reset raw data buffer counter
            self.accumulated_samples = 0
        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]
        return processed_samples if processed_samples != 0 else self.accumulated_samples

    def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1):
        if start_ndx != -1:
            end_ndx = start_ndx + int(n_feature_frames) \
                if start_ndx + n_feature_frames != 0 else len(self.feature_buffer)
            return self.feature_buffer[start_ndx:end_ndx, :][None, ].astype(np.float32)
        else:
            return self.feature_buffer[int(-1*n_feature_frames):, :][None, ].astype(np.float32)
        
    def __call__(self, x): return self._streaming_features(x)

class Model():
    def __init__(self, wakeword_models: List[str] = [], inference_framework: str = "onnx", device: str = 'gpu', **kwargs):
        self.model_path = wakeword_models[0]  # Use the first model provided
        self.model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        import onnxruntime as ort
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = 1
        sessionOptions.intra_op_num_threads = 1
        self.model = ort.InferenceSession(self.model_path, sess_options=sessionOptions, providers=["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"])
        self.model_input_shape = self.model.get_inputs()[0].shape[1]
        self.model_output_shape = self.model.get_outputs()[0].shape[1]
        self.model_prediction_function = functools.partial(self.model.run, None)
        self.prediction_buffer = deque(maxlen=30)
        self.preprocessor = AudioFeatures(inference_framework=inference_framework, device=device, **kwargs)

    def predict(self, x: np.ndarray, patience: dict = {}, threshold: dict = {}):
        n_prepared_samples = self.preprocessor(x) # Retrieve input name for the model
        input_name = self.model.get_inputs()[0].name # Prepare the input as a dict {input_name: input_data}
        features = self.preprocessor.get_features(self.model_input_shape)
        prediction_input = {input_name: features}
        prediction = self.model_prediction_function(prediction_input)
        predictions = {}
        if self.model_output_shape == 1:
            predictions[self.model_name] = prediction[0][0][0]
        else:
            for int_label, cls in self.class_mapping.items():
                predictions[cls] = prediction[0][0][int(int_label)]
        if len(self.prediction_buffer) < 5:
            for cls in predictions.keys():
                predictions[cls] = 0.0
        self.prediction_buffer.append(predictions.get(self.model_name, 0.0))
        return predictions


