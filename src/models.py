import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights
from torchvision import transforms

class ConvNextClassifier(nn.Module):
    def __init__(self, num_classes, model_size, pretrained=True, freeze_backborn=True):
        super().__init__()

        if model_size=='tiny':
            weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model=convnext_tiny(weights=weights)
        if model_size=='small':
            weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            self.model=convnext_small(weights=weights)
        if model_size=='base':
            weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            self.model=convnext_base(weights=weights)
        
        if model_size == 'tiny':
            feature_dim = 768
        elif model_size == 'small':
            feature_dim = 768
        elif model_size == 'base': 
            feature_dim = 1024

        in_features= self.model.classifier[2].in_features
        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)   
        )
        if freeze_backborn:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name: 
                    param.requires_grad = False  

    def forward(self, x):
        x = self.model(x)
        return x

    def unfreeze_layers(self, num_layers):
        total_stages=4
        stages_to_unfreeze=min(num_layers, total_stages)

        for name, param in self.model.classifier.parameters():
            param.requires_grad= True

        for i in range(total_stages-stages_to_unfreeze, total_stages):
            for param in self.model.features[i].parameters():
                param.requires_grad= True

def get_transforms(img_size=224, is_training=True):
    transform_list = []
    if is_training:
        transform_list.extend([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
        ])
    else:
        transform_list.extend([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(transform_list)
