import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	image = cv2.imread(current_path)
	#images.append(image)
	measurement = float(line[3])
	#measurements.append(measurement)
	correction=0.2
	measurement_left=measurement+correction
	measurement_right=measurement-correction
	image_left=cv2.imread('./data/IMG/'+line[1].split('/')[-1])
	image_right=cv2.imread('./data/IMG/'+line[2].split('/')[-1])
	images.extend((image,image_left,image_right))
	measurements.extend((measurement,measurement_left,measurement_right))
	

aug_images=[]
aug_measurements=[]

for image,measurement in zip(images,measurements):
	aug_images.append(image)
	aug_measurements.append(measurement)
	image_flipped = np.fliplr(image)
	measurement_flipped = -measurement
	aug_images.append(image_flipped)
	aug_measurements.append(measurement_flipped)
	#aug_images.append(cv2.flip(image,1))
	#aug_measurements.append(measurements*(-1))
	
X_train = np.array(aug_images)
Y_train = np.array(aug_measurements)


print("Number of training examples =", len(X_train));

print("Shape of each img", X_train[0].shape); 

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')


