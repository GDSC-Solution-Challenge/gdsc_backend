import tensorflow as tf
import efficientnet.keras as efn
import sys
import keras.applications.xception as xception

if len(sys.argv) != 2:
    print(f"Usage : {sys.argv[0]} h5_file")
    exit(0)

file_name = sys.argv[1]
model = tf.keras.models.load_model(file_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open( file_name.split('.')[0] + ".tflite", "wb").write(tflite_model)

