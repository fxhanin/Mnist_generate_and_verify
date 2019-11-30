# coding: utf8
# !/usr/bin/env python3
#
#-------------------------------------------------------------------
#  (c) François-Xavier Hanin 29/08/2019.
#  permanent dropout implemented as suggested by fchollet:
#  https://github.com/keras-team/keras/issues/1606
#
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#* Neither the name of François-Xavier Hanin nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#-------------------------------------------------------------------


import time
import numpy as np
from keras.datasets.mnist import load_data
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization
from keras import backend as K
from keras.layers.core import Lambda

# Starting to get a decent Wise Man to label images
def PermaDropout(rate):
	return Lambda(lambda x: K.dropout(x, level=rate)) # found at https://github.com/keras-team/keras/issues/1606#issuecomment-177595676

def Wise_Old_Man(dropRate=0.4):
	model = Sequential()

	# convolutions episode one
	model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)))
	model.add(BatchNormalization())
	model.add(PermaDropout(rate=dropRate))

	# convolutions episode two
	model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(PermaDropout(rate=dropRate))

	# Flat
	model.add(Flatten())

	# Final perceptrons layers
	model.add(Dense(192, activation='relu'))
	model.add(PermaDropout(rate=dropRate))
	model.add(Dense(10, activation='softmax'))

	# compile model
	opt = Adam(lr=0.0002)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	return model

# The Wise_Old_Man was trained on the MNIST train set.
WiseWeights = './Dropouts_Wise_old_Men/Aristotle_Weights_light.h5'
Aristotle = Wise_Old_Man()
Aristotle.load_weights(WiseWeights)

# Each class has its own GAN
generator_file = './GAN/generator_model.json'
list_of_weights = ['./GAN/generator_model_0.h5','./GAN/generator_model_1.h5','./GAN/generator_model_2.h5','./GAN/generator_model_3.h5','./GAN/generator_model_4.h5','./GAN/generator_model_5.h5','./GAN/generator_model_6.h5','./GAN/generator_model_7.h5','./GAN/generator_model_8.h5', './GAN/generator_model_9.h5']

json_file = open(generator_file, 'r')
loaded_model_json = json_file.read()
json_file.close()

Generators = []

for i in range(10):
	Generator = model_from_json(loaded_model_json)
	Generator.load_weights(list_of_weights[i])
	Generators.append(Generator)
# All Set.

# Let's start generating
number_generated = 0
number_ambiguous = 0

numberOfImagesToGenerate = 6000 # number of images to generate for each class

for j in range(10):
	currentNumber = j
	currentArray = np.empty((numberOfImagesToGenerate,28,28,1))

	for h in range(numberOfImagesToGenerate):
		found = False
		while not found:
			# generate an image
			latency = np.random.randn(100).reshape(1, 100)
			CurrentImage = Generators[j].predict(latency)
			number_generated += 1

			# Get 100 advices from Aristotle about this
			theMatrix = np.empty((100,10))
			for k in range(100):
				AristotleAdvice = Aristotle.predict(CurrentImage)
				theMatrix[k] = AristotleAdvice[0][:]

			# Get some stats
			AristotleMean = np.round(np.mean(theMatrix, axis=0), decimals=3)
			AristotleStd = np.round(np.std(theMatrix, axis=0), decimals=3)
			IndicesFomMax2Min = (-AristotleMean).argsort()[:10]
			WiseAnswer = IndicesFomMax2Min[0] # number getting the max value. Considered as the WiseMan answer
			WiseChalenger = IndicesFomMax2Min[1] # if the old man hesitates, well, it's with this value

			# 95% confidence intervals
			First95CIlow, First95CIhigh = (AristotleMean[WiseAnswer]-(2*(AristotleStd[WiseAnswer])), AristotleMean[WiseAnswer]+(2* (AristotleStd[WiseAnswer])) )
			Second95CIlow, Second95CIhigh = (AristotleMean[WiseChalenger]-(2*(AristotleStd[WiseChalenger])), AristotleMean[WiseChalenger]+(2* (AristotleStd[WiseChalenger])) )

			# store only if the generated number is correctly identified
			if WiseAnswer == currentNumber: # the wise man gave the right answer
				if Second95CIhigh >= First95CIlow: # but he hesitates with the second
					currentArray[h,:,:,:] = CurrentImage[0,:,:,:] # so we store the ambiguous image
					found = True
					number_ambiguous += 1
					print(time.strftime("%Y%m%d_%H%M%S"), 'for', j, 'the', h, 'image was found. Hesitates with',WiseChalenger,'. Ambiguous',str(number_ambiguous), '/',str(number_generated))
				else:
					# The image is probably clear-cut with no ambiguous shape.
					currentArray[h,:,:,:] = CurrentImage[0,:,:,:] # so we store the clear cut image - if we want to 
					found = True
					print(time.strftime("%Y%m%d_%H%M%S"), 'for', j, 'the', h, 'image was found. No Hesitation.', str(number_generated), 'generated')

	# save the arrays
	answers = np.ones((numberOfImagesToGenerate,1))*currentNumber
	savename = time.strftime("%Y%m%d_%H%M%S") + '_generated_' + str(currentNumber) + '.npz'
	np.savez(savename, X=currentArray, Y=answers)



