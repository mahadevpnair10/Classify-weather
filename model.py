
import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes: int):
  
    """Creates an EfficientNetB2 model."""
   
    # Create model and transforms
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )
    
    return model, transforms
