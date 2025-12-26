import torch 
from torch.utils.data import Dataset 
from torchvision import datasets 
from torchvision.transforms import ToTensor 
import matplotlib.pyplot as plt 

# creating a custom dataset  
import os 
import pandas as pd 
from torchvision.io import decode_image 

from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(
    root = "data", 
    train = True, 
    download = True, 
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data", 
    train = False, 
    download = True, 
    transform = ToTensor()
)

labels_map = {
    0: "T-Shirt", 
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize = (8, 8))
cols, rows = 3, 3 
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size = (1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = "grey")
plt.show()

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir 
        self.transform = transform 
        self.target_transform = target_transform 
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # the decode_image() generates a tensor from an image 
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

# display the image and label 
train_features, train_labels = next(iter(train_dataloader))
# the next method is only defined for iterators and not iterables 
# the iterator is like a bookmark and that's what you create with the dataloader which currently is an iterable 
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap = "gray")
plt.show()
print(f"Label: {label}")

# usually the data is not in the format that we want 
# aka we usually want the features to be normalized tensors 
# the labels as one hot encoded vectors 

# the transform() function is usually just a ToTensor()
# the target_transform is a Lambda which does turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y


from torch import nn 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits 
        
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
# never call model.forward() directly 

# dim = 0 always corresponds to the batch that you are doing 
# this is the number of images that you are processing at once 

# channels = 1 for GrayScale and 3 for RGB 
x = torch.rand(1, 28, 28, device=device)
logits = model(x)
# the logits are just raw numbers 
pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)

# the 1 here for the argmax is to get the value per image? 
print(f"Predicted class: {y_pred}")

# .flatten() always goes from dim=1 onwards 
# assumes dim = 0 is the batch dim 

input_image = torch.rand(3, 28, 28)
flatten = nn.Flatten()
flat_image = flatten(input_image)

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)

hidden1 = nn.ReLU()(hidden1)
# you create the ReLU() module first 
# can't do nn.ReLU(hidden1)

# The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the nn.Softmax module 

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")




