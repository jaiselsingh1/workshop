import torch 
from torch import nn 
from torch.utils.data import DataLoader 
# dataloader creates an iterable over the dataset (DataLoader and Dataset are the two main primitives around which torch is built)
from torchvision import datasets
from torchvision.transforms import ToTensor 

# each dataset features 2 main arguments within PyTorch 
# transform and target transform to label and modify the samples/data accordingly 
# transform applied to the input images 
# target transform applied to the labels/target (this is to usually convert labels to diff formats)

# target transform has one use case for one hot encoding where you convert categorical data into vectors where one is on and one is off aka the 1/0 

# download the training data from the open sources 
training_data = datasets.FashionMNIST(
    root = "data", 
    train = True, 
    download = True, 
    transform = ToTensor(), 
)

# download the test data 
test_data = datasets.FashionMNIST(
    root = "data", 
    train = False, 
    download = True, 
    transform = ToTensor(), 
)

# wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading
# this is the number of samples that you process before you do any weight updates
batch_size = 64 

# create the data loaders 
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")

# batch sizes can be the size of a replay buffer vs the mini batches that you run for PPO for instance 
# replay buffer size (bigger batch) == collect these many transitions and then we can do the mini updates using the smaller batch size 

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"using device:{device}")

# define the model 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack == nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)