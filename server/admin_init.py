import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials
import tensorflow as tf

firebase_admin.initialize_app(
  credentials.Certificate("C:/Develop/GDSC/gdsc_backend/server/cycleye-948b4-firebase-adminsdk-8apjo-cc46143c30.json"),
  options={
      'storageBucket': "cycleye-948b4.appspot.com",
  })

# First, import and initialize the SDK as shown above.

# Load a tflite file and upload it to Cloud Storage
# model = tf.keras.models.load_model("C:/Develop/GDSC/gdsc_backend/models/baseOX_230329.h5")
# source = ml.TFLiteGCSModelSource.from_keras_model(model)
source = ml.TFLiteGCSModelSource.from_tflite_model_file('C:/Develop/GDSC/gdsc_backend/models/baseOX_230329.tflite')


# Create the model object
tflite_format = ml.TFLiteFormat(model_source=source)
model = ml.Model(
    display_name="baseOX",  # This is the name you use from your app to load the model.
    tags=["baseOX"],             # Optional tags for easier management.
    model_format=tflite_format)

# Add the model to your Firebase project and publish it
new_model = ml.create_model(model)
ml.publish_model(new_model.model_id)