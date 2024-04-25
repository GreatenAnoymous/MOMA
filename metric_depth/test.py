import torch
import torchvision.transforms as transforms

x = torch.randn(3, 224, 224, device="cuda:1")
out = transforms.Resize(1000)(x)