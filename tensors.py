import torch 
import numpy as np 

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# they can be inferred from data or even just converted from numpy arrays 
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"ones tensor {x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"random tensor {x_rand}")

# shape is a tuple for the dimensions of the tensor 
shape = (2, 3, )
rand_tensor = torch.rand(shape)
shape2 = (2, 3)
ones_tensor = torch.ones(shape)
if (ones_tensor == torch.ones(shape2)).all():
    print("its the same")
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# tensors have various different attributes that are associated with them 
# they have a shape, dtype, and a .device for the device that they are stored on 

# tenors are inherently first created on the CPU and then they can be moved to an "accelerator"
# this is done using the .to() method 

# if torch.accelerator.is_available():
#     tensor = tensor.to(torch.accelerator.current_accelerator())

# there is torch.cat() and torch.stack()
tensor = torch.ones(4,4)
tensor[:, 1] = 0.0 
print(tensor)

new_tensor = torch.cat([tensor, tensor], dim=1)
print(new_tensor)

# torch.cat() concatenates along a direction that already exists 
# torch.stack() will create a new dimension to stack along 

# Use cat() when you want to extend an existing dimension (like appending to a list)
# Use stack() when you want to batch tensors together or create a new axis (like creating a batch dimension for training)

