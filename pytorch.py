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
            # the ReLU activation function is defined as being f(x) = max(0, x)
            # the tanh() activation would bound it between -1 and 1 for the ouptput
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    # logits are raw numbers that are not converted into probabilities by being passed into a softmax function (or some other activation for the final layer)
    # the reLU is the activation in the middle for learning non linearities 

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    # in the training loop, the model does predictions on the training dataset and then backprops the pred error
    # this error is used to make updates to the params in the batch_size 
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # compute the prediction error 
        pred = model(x)
        loss = loss_fn(pred, y)

        # backprop 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

