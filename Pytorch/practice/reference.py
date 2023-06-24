import torch
import pandas as pd
import numpy as np
import matplotlib as plt

print("Torch: " + str(torch.__version__))
print("Numpy: " + str(np.__version__))
print("Pandas: " + str(pd.__version__))
print("MatPLotLib: " + str(plt.__version__) + "\n\n\n\n")

#! Manually creating tensors
tensor1 = torch.tensor([1, 2, 3])

#! Tensor attributes
print(f"Tensor: {tensor1}, shape: {tensor1.shape}, dtype: {tensor1.dtype}, device: {tensor1.device}, dimensions: {tensor1.ndim}")

tensor2 = torch.tensor([[1, 2, 3],
                       [3, 4, 5]])
print(f"Tensor: {tensor2}, shape: {tensor2.shape}, dtype: {tensor2.dtype}, device: {tensor2.device}, dimensions: {tensor2.ndim}")

tensor3 = torch.tensor([[[1, 2, 3],
                       [3, 4, 5]],

                       [[6, 7, 8],
                       [9, 10, 11]]])
print(f"Tensor: {tensor3}, shape: {tensor3.shape}, dtype: {tensor3.dtype}, device: {tensor3.device}, dimensions: {tensor3.ndim}\n\n\n\n")

#! Transposing a matrix
print(tensor2.T)
print(f"{tensor2.T.shape}\n\n\n\n")

#! Matrix Multiplication - Dot Product
print(tensor2.matmul(tensor2.T))
print(f"Original: {tensor2.dtype}, New: {tensor2.to(torch.float64).dtype}\n\n\n\n")

#? Matrix addition and the "to" operator (changes attributes)
#! Note: the "to" operation only changes the attribute temporarily
print(f"Matrix Addition: {tensor2.to(torch.float16) + tensor2}")
print(f"{tensor2.dtype}\n\n\n\n")

#! Creating Random Matrices
print(f"Random Matrix: {torch.rand(2, 2, 3)}")
print(f"Matrix Ones: {torch.ones(2, 2, 3)}")
print(f"Matrix Zeros: {torch.zeros(2, 2, 3)}\n\n\n\n")

#! The "_like" operation, creates tensor with same dimensions
someInput = torch.rand(4, 2, 2)
print(f"Matrix Same size as some Input: {torch.rand_like(someInput)}")
print(f"Matrix Zeros Same size as some Input: {torch.zeros_like(someInput)}")
print(f"Matrix Ones Same size as some Input: {torch.ones_like(someInput)}\n\n\n\n")