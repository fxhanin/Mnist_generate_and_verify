# Mnist_generate_and_verify
A simple script to verify the GAN-generated mnist images using a CNN with permanent dropout

Generate & label MNIST images with class-specific GAN and permanent dropout

1. Background.
The training of CNNs using available datasets has become very easy to perform using several freely available libraries. Performance of such CNNs may rise very easily and quickly up to 99,2 % on a simple laptop. 
However, the gap between 99,2 and 99,8 % accuracy is much harder to cross. 

Several methods have been described elsewhere to reach such performance. The main errors on the test dataset arise from ambiguous images, where even a human eye could misinterpret a 3 for a 5, or a 0 for a 8. 

The pupose of this work was to assess if it is possible to generate automaticaly hundred or thousands of new images, and label them in the same movement.

We will first use a GAN to generate MNIST images, but by creating a GAN for each class.
Then assess the truth of the label using a wise CNN obtaining error bars on the answer with permanent dropout.

2. train GANs for each class.
We will train 10 different GANs, one for each class. 
There are lots of keras code for MNIST GANs on the web, so I will not reproduce the code here. I used the very clear code from Jason Brownlee (MachinelearningMastery.com); except for the dataset selection, and with a loop to train 10 different GANs, one for each class.
There are numerous other models out there, feel free to use any other you prefer. 

I defined a Extract_Class_from_MNIST(anumber) function to extract only the images of a specific number in the MNIST train dataset.

Then we train 10 different GANs, one for each class and save the specific weights for each class. Weights as h5 hiles & model as json file are provided on the GAN folder.

3. Generate and verify.
Now that we have 10 different GANs we can use them to generate specific digits, knowing in advance to which class they must belong.

But we need to check that. For that purpose, we'll use permanent dropout in a CNN model trained to recognize digits on the MNIST dataset.

The model is quite simple, but dropout layers will be permanently droped out, even at prediction time.

When the prediction is required, we will not predict only once, but hundred times, and the permanent dropout wil randomly neutralize perceptrons each time. In this way, we will obtain 100 different answers.

The 100 different answers will then be processed for mean and standard deviation on each class using the activation values from output.

If one class gets a mean of 0,999 and a standard deviation of zero, well, it means that the network is very sure of the answer.
On the other hand, when class A gets for instance 0.67 +- 0.2 ; and class B 0.51 +- 0.3 ; it means thats sometimes A gets the highest activation, sometimes B. The overlap between values (I used the 95% confidence interval overlap, i.e. 2 SDs) means that the network hesitates between A and B.

script 02_GANs_generate_AND_verify.py does the job, saving each time 6000 images generated from each class-specific GAN.

The code allows to select only ambiguous images if you wish to get only ambiguous images. It takes a while thought. Only one in maybe 30 or 40 images, depending on the GAN, is considered as ambiguous, and remember, each image is predicted 100 times, before performing statistics on the results.  


