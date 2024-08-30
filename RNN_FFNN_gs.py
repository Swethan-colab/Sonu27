import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(100, 100).to(device)
print(device)