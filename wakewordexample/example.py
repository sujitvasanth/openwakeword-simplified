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
