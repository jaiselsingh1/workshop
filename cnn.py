# input * kernel = output matrix 
# ouput is from sliding the kernel on the input 

# size of output = size of input - size of kernel + 1 
# cross correlation != convolution 
# for convolution you rot(180) on the kernel?

# star is cross correlation 
# asterix is convolution 

# valid correlation vs full correlation 
# valid is when there's an entire fit for the K on the input 
# full is when there's any overlap and hence the output size is larger too 

# kernels must extend the full depth of the input 
# each kernel has a bias matrix the size of the output 
# depth of the output == number of kernels 

# y1 = bias + input1 * kernel11 + input2 * kernel12
# feature depth == number of channels at each given layer 
# The dense layer is just a specific example of the general convolution algorithm 

import numpy as np  
from scipy import signal 

class Layer:
    def __init__(self):
        self.input = None 
        self.output = None
    def forward(self, input):
        pass
    def backward(self, output_gradient, learning_rate):
        pass

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape 
        self.depth = depth 
        self.input_shape = input_shape
        self.input_depth = input_depth 
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        # input depth == depth of each kernel?
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        # the * there does unpacking for the argument shape 
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input 
        self.output = np.copy(self.biases)
        # think of depth of as the number of "parallel representations" of the same input 
        # depth == number of kernels
        # (depth, spatial size) -> you can do pooling where the depth stays the same but the spatial size decreases

        # nested for loop for going through the depth and then the input depth 
        for i in range(self.depth):
            # for each ouput channel, you go through the kernels (aka self.input_depth and do the convolution)
            for j in range(self.input_depth):
                # not commutative; aka the order matters and the "valid" is the type we are doing, not doing "full" for instance
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output 
    
    # self.input_depth = the number of channels coming in 
    # self.depth = the number of channels coming out 

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

            self.kernels -= learning_rate * kernels_gradient
            self.biases -= learning_rate * output_gradient
        return input_gradient



