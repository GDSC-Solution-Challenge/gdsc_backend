import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials

# First, import and initialize the SDK as shown above.
firebase_admin.initialize_app(
  credentials.Certificate("C:/Develop/GDSC/gdsc_backend/server/cycleye-948b4-firebase-adminsdk-8apjo-cc46143c30.json"),
  options={
      'storageBucket': "cycleye-948b4.appspot.com",
  })

face_detectors = ml.list_models().iterate_all()
print("Find models:")
for model in face_detectors:
  print('{} (ID: {})'.format(model.display_name, model.model_id))