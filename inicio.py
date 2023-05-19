import cv2 # OpenCV
import numpy as np


imagem = cv2.imread('/content/person.jpg')


imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


detector_facial = cv2.CascadeClassifier('content/haarcascade_frontalface_default.xml')

deteccoes = detector_facial.detectMultiScale(imagem_cinza)

deteccoes

len(deteccoes)

for (x, y, w, h) in deteccoes:
	cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 4)
cv2_imshow(imagem)