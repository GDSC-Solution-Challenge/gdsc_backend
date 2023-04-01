from keras.models import load_model
import keras.applications.xception as xception
from PIL import Image
import numpy as np

# 이미지를 로드하고 크기를 조정합니다.
img = Image.open("testimage2.jpg").resize((224, 224))
# 이미지를 넘파이 배열로 변환합니다.
x = np.array(img)
# 이미지를 모델이 예상하는 크기와 형태로 변경합니다.
x = x.reshape((1,) + x.shape)

model = load_model('classification_model.h5')

predict = model.predict(x)[0]
for i, p in enumerate(predict):
    print("Class {}: {:.2f}%".format(i, p * 100))
