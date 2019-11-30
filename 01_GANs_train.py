# coding: utf8
# !/usr/bin/env python3
#
#-------------------------------------------------------------------
#  This code uses functions from machinelearningmastery.com
#  https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
#  (c) Jason Brownlee

#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#* Neither the name of François-Xavier Hanin nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#-------------------------------------------------------------------

from numpy import expand_dims
from numpy import zeros
from keras.datasets.mnist import load_data

from machinelearningmastery.com import define_discriminator, define_generator, define_gan, train

def Extract_Class_from_MNIST(anumber):
	(X_train, Y_train), (X_test, Y_test) = load_data()
	indices = [i for i, x in enumerate(Y_train) if x == anumber]
	anarrayX = zeros((len(indices), 28, 28))
	for i in range(len(indices)):
		anarrayX[i,:,:] = X_train[indices[i],:,:]
	X = expand_dims(anarrayX, axis=-1)
	X = X.astype('float32')
	X = X / 255.0
	return X

for k in range(10):
	# This code is from machinelearnigmastery.com except for the dataset selection
	# check https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/ for the functions
	latent_dim = 100
	d_model = define_discriminator()
	g_model = define_generator(latent_dim)
	gan_model = define_gan(g_model, d_model)
	dataset = Extract_Class_from_MNIST(k)
	train(g_model, d_model, gan_model, dataset, latent_dim)

