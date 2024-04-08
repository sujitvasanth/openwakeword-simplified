# openwakeword-simplified
This is a simplified version of openwakeword https://github.com/dscripka/openWakeWord

You should have example folder with 
* model.py                        the library
* melspectrogram.onnx             NN to calculate mel spectrogram
* embedding_model.onnx            NN to calculate audio features
* your_wakeword.onnx              custom NN for your wakeword
* your_pythonfile.py              your code to run inference

sample code to run inference is here
<pre lang=LANG>
import pyaudio, numpy as np, model
model = model.Model(wakeword_models=["Eugene.onnx"],inference_framework ="onnx")
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,
    input=True, frames_per_buffer=3000)
while True:
    audio = np.frombuffer(mic_stream.read(3000), dtype=np.int16)
    prediction = model.predict(audio)
    if prediction["Eugene"] >0.11:
        print(" ============== Wakeword Detected ==============")
        print(prediction["Eugene"]);print(model.predict(audio));
        print(model.predict(audio));print(model.predict(audio))</pre>

Requires onnxruntime with gpu support but will work on cpu (need to change device in model.py) 
<pre lang=LANG>
pip install onnxruntime-gpu==1.17.0 --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
</pre>



