import torch 

#the most frequently used algorithm is back propagation. In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter

x = torch.ones(5)
y = torch.zeros(3)
 
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# can later also do something like x.requires_grad_(True) method 

z = torch.matmul(x, w) + b 

# this order is due to batch processing?
# usually you do x @ w because the x dims first have the batch for the outer dim hence there will be a dim mismatch 
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# X @ W = Z
# (batch, in_features) @ (in_features, out_features) = (batch, out_features)

# all tensors with requires_grad=True are tracking their computational history and support gradient computation
# however, often you might need to disable this once the model has been trained 
# this is done by surrounding it with a torch.no_grad() block 

with torch.no_grad():
    z = torch.matmul(x, w) + b 
print(z.requires_grad)

# another way to get the same result is to use the .detach() method on the tensor 
z = torch.matmul(x, w) + b 
z_det = z.detach()
# To mark some parameters in your neural network as frozen parameter 
# To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient 

# View autograd as an DAG 
# leaves are input tensors (nothing that is dependent on them)
# roots are output tensors (no parents)

# forward pass -> does the computation to compute the resulting tensor + maintains operation's gradient function 
# .backward() computed on the root 
# computes gradient from .grad_fn() and accumulates into the .grad attribute 
# chain rule to propogate down to the leaves 

# instead of computing the Jacobian matrix, pytorch allows you to compute the jacobian vector product 
# so instead of J you get v^T where v is fed in as an input 

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()

# into the backward function you have to give the vector 
# the vector will be the same size as the ouput
# think of d(loss)/dx1, d(loss)/dx2, ... 
# here the out is the loss function
out.backward(torch.ones_like(out), retain_graph = True)
# the vector possed in is how much you "care" about each element of the output 

inp.grad.zero_()
#  PyTorch accumulates the gradients, i.e. the value of computed gradients is added to the grad property of all leaf nodes of computational graph
# this is mega useful when your GPU isn't enough so you run mini batches 

# gradient of a sum == sum of gradients 


