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
from keras.datasets import mnist
import keras  # Updated import
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Base layer class
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
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        # the bias gradient == output_gradient itself which is an input 
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):  
        self.input_shape = input_shape 
        self.output_shape = output_shape 

    def forward(self, input):
        # reshapes the input to the ouput shape
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        # reshapes the output back to the input shape
        return np.reshape(output_gradient, self.input_shape)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred + 1e-7) - y_true / (y_pred + 1e-7)) / np.size(y_true)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation 
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input 
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = keras.utils.to_categorical(y)  # Fixed import
    y = y.reshape(len(y), 2, 1)
    return x, y

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
    
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
    error_history = []
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        error_history.append(error)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error:.4f}")
    
    return error_history

def visualize_kernels(conv_layer, title="Learned Kernels"):
    """Visualize the learned convolutional kernels"""
    kernels = conv_layer.kernels
    depth, input_depth, k_h, k_w = kernels.shape
    
    fig, axes = plt.subplots(depth, input_depth, figsize=(input_depth*2, depth*2))
    fig.suptitle(title, fontsize=16)
    
    # Handle different cases for axes array shape
    if depth == 1 and input_depth == 1:
        axes = np.array([[axes]])
    elif depth == 1:
        axes = axes.reshape(1, -1)
    elif input_depth == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(depth):
        for j in range(input_depth):
            ax = axes[i, j]
            kernel = kernels[i, j]
            im = ax.imshow(kernel, cmap='coolwarm', vmin=-2, vmax=2)
            ax.set_title(f'K[{i},{j}]', fontsize=8)
            ax.axis('off')
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    return fig

def visualize_feature_maps(conv_layer, input_image, title="Feature Maps"):
    """Visualize the feature maps produced by a convolutional layer"""
    output = conv_layer.forward(input_image)
    depth = output.shape[0]
    
    cols = min(5, depth)
    rows = (depth + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle(title, fontsize=16)
    
    axes = axes.flatten() if depth > 1 else [axes]
    
    for i in range(depth):
        axes[i].imshow(output[i], cmap='viridis')
        axes[i].set_title(f'Channel {i}', fontsize=10)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(depth, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_predictions(network, x_test, y_test, num_samples=10):
    """Visualize predictions vs ground truth"""
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(x_test))):
        output = predict(network, x_test[i])
        pred = np.argmax(output)
        true = np.argmax(y_test[i])
        
        axes[i].imshow(x_test[i][0], cmap='gray')
        color = 'green' if pred == true else 'red'
        axes[i].set_title(f'Pred: {pred}, True: {true}', color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # load MNIST from server, limit to 100 images per class since we're not training on GPU
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 100)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

    # neural network
    network = [
        Convolutional((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]

    print("Starting training...")
    # train
    error_history = train(
        network,
        binary_cross_entropy,
        binary_cross_entropy_prime,
        x_train,
        y_train,
        epochs=20,
        learning_rate=0.1
    )

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(error_history, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./vis/training_curve.png', dpi=150)
    print("Saved training_curve.png")
    plt.show()

    # Visualize learned kernels
    conv_layer = network[0]
    fig = visualize_kernels(conv_layer, "Learned Convolutional Kernels")
    plt.savefig('./vis/learned_kernels.png', dpi=150)
    print("Saved learned_kernels.png")
    plt.show()

    # Visualize feature maps for a sample image
    sample_image = x_test[0]
    fig = visualize_feature_maps(conv_layer, sample_image, "Feature Maps for Test Image")
    plt.savefig('./vis/feature_maps.png', dpi=150)
    print("Saved feature_maps.png")
    plt.show()

    # Visualize predictions
    fig = visualize_predictions(network, x_test, y_test, num_samples=10)
    plt.savefig('./vis/predictions.png', dpi=150)
    print("Saved predictions.png")
    plt.show()

    # test accuracy
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    correct = 0
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        pred = np.argmax(output)
        true = np.argmax(y)
        if pred == true:
            correct += 1
        print(f"pred: {pred}, true: {true}")
    
    accuracy = correct / len(x_test) * 100
    print("="*50)
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(x_test)})")
    print("="*50)


if __name__ == "__main__":
    main()