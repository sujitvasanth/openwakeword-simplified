# openwakeword-simplified
simplified version of openwakeword with only 1 python file and 2 onnx files

this is a simplified version of https://github.com/dscripka/openWakeWord

You should have 1 folder with 
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
mic_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=3000)

while True:
    audio = np.frombuffer(mic_stream.read(3000), dtype=np.int16)
    prediction = model.predict(audio)
    #print(prediction)
    if prediction["Eugene"] >0.11:
        print(" ============== Wakeword Detected ==============")
        print(prediction["Eugene"])
        print(model.predict(audio))
        print(model.predict(audio))
        print(model.predict(audio))
</pre>
  




