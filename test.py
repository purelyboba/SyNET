import torch
import torch.nn as nn

FILE = "pretrained_model.pth"

model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)
