from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow as tf
from pydub import AudioSegment
import numpy as np

app = Flask(__name__)

def getSpectrogram(audio):

    rightSamples = 16000
    numMelBins = 40

    audio = tf.cast(audio, tf.float32) / 32768.0
    audio = audio[:rightSamples]
    audio = tf.pad(audio, [[0, rightSamples - tf.shape(audio)[0]]])
    
    stft = tf.signal.stft(audio, frame_length=640, frame_step=320)

    spectrogram = tf.abs(stft)

    numSpectrogramBins = spectrogram.shape[-1]
    

    melMatrix = tf.signal.linear_to_mel_weight_matrix(numMelBins, numSpectrogramBins, rightSamples)

    melSpectrogram = tf.tensordot(spectrogram, melMatrix, 1)

    melSpectrogram.set_shape(spectrogram.shape[:-1].concatenate([numMelBins]))

    logMelSpectrogram = tf.math.log(melSpectrogram + 1e-6)
    logMelSpectrogram = logMelSpectrogram[..., tf.newaxis]
    logMelSpectrogram.set_shape((49, 40, 1))
    return logMelSpectrogram

def loadAudio(filePath, target_sample_rate=16000):
    audio = AudioSegment.from_file(filePath)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(target_sample_rate)
    samples = np.array(audio.get_array_of_samples())
    tensorAudio = tf.convert_to_tensor(samples, dtype=tf.float32)
    return tensorAudio

equiv = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', '_silence_', '_unknown_']
cnnModel = keras.models.load_model("models/0.95CNN.keras")

@app.route("/cnn/predict", methods=["POST"])
def predictWithCnn():
    if 'file' not in request.files:
        return jsonify({"error": "No se recibió el archivo enviado."})
    
    audio = request.files["file"]
    audio.save("aux.wav")

    spectrogram = []
    spectrogram.append(getSpectrogram(loadAudio("aux.wav")))
    spectrogram = tf.stack(spectrogram) 
    prediction = cnnModel.predict(spectrogram)
    prediction = np.argmax(prediction, axis=1)
    prediction = [equiv[i] for i in prediction]
    print(prediction)
    return jsonify({"msg": prediction[0]})


if (__name__ == "__main__"):
    
    app.run(host="0.0.0.0", port=6000, debug=True)