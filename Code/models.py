"""Model definitions for federated learning."""
import torch
import torch.nn as nn
import torchvision.models as models


class BaseModel(nn.Module):
    def replace_batchnorm_with_groupnorm(self):
        """Replace all BatchNorm layers with GroupNorm."""
        for name, module in list(self.named_modules()):  # Use list() to avoid dictionary size errors
            if isinstance(module, nn.BatchNorm2d):
                num_features = module.num_features
                new_module = nn.GroupNorm(num_groups=8, num_channels=num_features)
                # Replace the module at the correct place
                parent_module = self._get_parent_module(name)
                setattr(parent_module, name.split('.')[-1], new_module)

    def _get_parent_module(self, module_name):
        """Get the parent module given the full name of the module."""
        names = module_name.split('.')
        module = self
        for name in names[:-1]:
            module = getattr(module, name)
        return module


# ResNet50 model
class NetResNet(BaseModel):
    def __init__(self):
        super(NetResNet, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Adjust for 10 classes (CIFAR-10)

    def forward(self, x):
        return self.model(x)


# EfficientNetV2 model
class NetEfficientNetV2(BaseModel):
    def __init__(self):
        super(NetEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 10)  # Adjust for 10 classes

    def forward(self, x):
        return self.model(x)


# MobileNetV3 model
class NetMobileNetV3(BaseModel):
    def __init__(self):
        super(NetMobileNetV3, self).__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 10)  # Adjust for 10 classes

    def forward(self, x):
        return self.model(x)