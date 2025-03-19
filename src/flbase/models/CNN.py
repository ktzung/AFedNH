from ..model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from .ResNet import ResNet18, ResNet18NoNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvICHPretrained(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config['num_classes']
        self.return_embedding = config.get('return_embedding', False)
        # Load a pretrained ResNet (e.g., ResNet-18)
        # Use pretrained='imagenet' for ImageNet weights
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept the correct number of input channels
        # if config.get('in_channels', 3) != 3:  # Only modify if not 3 channels
        self.base_model.conv1 = nn.Conv2d(config.get('in_channels', 3), 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer with our own
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, self.num_classes)


    def forward(self, x):
        if self.return_embedding:
          x = self.base_model.conv1(x)
          x = self.base_model.bn1(x)
          x = self.base_model.relu(x)
          x = self.base_model.maxpool(x)

          x = self.base_model.layer1(x)
          x = self.base_model.layer2(x)
          x = self.base_model.layer3(x)
          x = self.base_model.layer4(x)

          x = self.base_model.avgpool(x)
          embedding = torch.flatten(x, 1)
          logits = self.base_model.fc(embedding)
          return embedding, logits
        else:
          return self.base_model(x)

class ConvICHDeeper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.get('in_channels', 3)
        self.num_classes = config['num_classes']
        self.return_embedding = config.get('return_embedding', False)

        # Increased number of convolutional blocks
        self.conv_block1 = self._make_conv_block(self.in_channels, 64)  # 3 -> 64
        self.conv_block2 = self._make_conv_block(64, 128)  # 64 -> 128
        self.conv_block3 = self._make_conv_block(128, 256) # 128 -> 256
        self.conv_block4 = self._make_conv_block(256, 512) # 256 -> 512


        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(config.get('dropout_rate', 0.5))
        self.fc2 = nn.Linear(256, self.num_classes)


    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        logits = self.fc2(x)

        if self.return_embedding:
            return x, logits  # Return before the final layer
        else:
            return logits


class Conv4ICH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.in_channels = config.get('in_channels', 3)  # Expect 3 channels
        self.num_classes = config['num_classes']  # Should be 6 for RSNA
        self.image_size = config['input_size'] #Keep for other usage
        self.return_embedding = config.get('return_embedding', False)

        # --- Four Convolutional Layers ---
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1) # Changed to 3x3 kernel
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Increased filters, 3x3 kernel
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling after two layers

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Increased filters, 3x3 kernel
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # Increased filters, 3x3 kernel
        self.bn4 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Pooling after two layers


        # Adaptive Pooling (keeps output size consistent)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        pooled_size = 5

        # Adjust linear layer input size (512 channels * pooled_size * pooled_size)
        self.linear1 = nn.Linear(512 * pooled_size * pooled_size, 384)
        self.dropout1 = nn.Dropout(config.get('dropout_rate', 0.5))
        self.linear2 = nn.Linear(384, 192)
        self.dropout2 = nn.Dropout(config.get('dropout_rate', 0.5))

        # Output layer (no sigmoid here if using BCEWithLogitsLoss)
        self.fc_out = nn.Linear(192, self.num_classes)

    def forward(self, x):
        # --- Convolutional Layers with ReLU and BatchNorm ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)  # Pooling after the first two layers

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)  # Pooling after the second two layers

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)

        logits = self.fc_out(x)

        if self.return_embedding: #Return the x before output
          return x, logits
        else:
          return logits

class Conv2ICH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.in_channels = config.get('in_channels', 3)  # Expect 3 channels after windowing
        self.num_classes = config['num_classes']  # Should be 6
        self.image_size = config['input_size']
        self.return_embedding = config.get('return_embedding', False)

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        pooled_size = 5

        self.linear1 = nn.Linear(64 * pooled_size * pooled_size, 384)
        self.dropout1 = nn.Dropout(config.get('dropout_rate', 0.5))
        self.linear2 = nn.Linear(384, 192)
        self.dropout2 = nn.Dropout(config.get('dropout_rate', 0.5))

        # Output layer with sigmoid (for multi-label)
        self.fc_out = nn.Linear(192, self.num_classes)
        #self.prototype = nn.Linear(192, config['num_classes'], bias=False)
        # --- Add prototype and scaling ---
        #temp = nn.Linear(192, self.num_classes, bias=False).weight.data
        #self.prototype = nn.Parameter(temp.clone())
        #self.scaling = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)

        logits = self.fc_out(x)
        # No Sigmoid here, apply it during loss calculation if using BCEWithLogitsLoss
        output = torch.sigmoid(logits)  # Use sigmoid for multi-label

        if self.return_embedding:  # Return x before the output layer
             return x, logits
        else:
            return logits
        #feature_embedding = F.relu(self.linear2(x))  # Store the embedding
        #x = self.dropout2(feature_embedding)

         # --- Normalize feature embedding ---
        #feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        #feature_embedding = torch.div(feature_embedding, feature_embedding_norm)

        # --- Normalize prototype weights (if trainable) ---
        #if self.prototype.requires_grad:
        #    prototype_norm = torch.norm(self.prototype, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        #    normalized_prototype = torch.div(self.prototype, prototype_norm)
        #else:
        #    normalized_prototype = self.prototype

         # --- Calculate logits using matrix multiplication and scaling ---
        #logits = self.scaling * torch.matmul(feature_embedding, normalized_prototype)

        #if self.return_embedding:
        #    return feature_embedding, logits
        #else:
        #    return logits

#Class Conv2ICH work with the FedProto
class Conv2ICHProto(Model):
    def __init__(self, config):
        super().__init__(config)
        self.in_channels = config.get('in_channels', 3)  # Expect 3 channels after windowing
        self.num_classes = config['num_classes']  # Should be 6
        self.image_size = config['input_size']
        self.return_embedding = config.get('return_embedding', False)

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        pooled_size = 5

        self.linear1 = nn.Linear(64 * pooled_size * pooled_size, 384)
        self.dropout1 = nn.Dropout(config.get('dropout_rate', 0.5))
        self.linear2 = nn.Linear(384, 192)
        self.dropout2 = nn.Dropout(config.get('dropout_rate', 0.5))

        # Prototype layer for FedProto
        self.prototype = nn.Linear(192, self.num_classes, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        feature_embedding = F.relu(self.linear2(x))
        x = self.dropout2(feature_embedding)

        logits = self.prototype(x)

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits

class Conv2Cifar(Model):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 53 * 53, 384)
        self.linear2 = nn.Linear(384, 192)
        # intentionally remove the bias term for the last linear layer for fair comparison
        self.prototype = nn.Linear(192, config['num_classes'], bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 53 * 53)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 53 * 53)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return x, logits


class Conv2CifarNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 53 * 53, 384)
        self.linear2 = nn.Linear(384, 192)
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 53 * 53)
        x = F.relu(self.linear1(x))
        feature_embedding = F.relu(self.linear2(x))
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


# class ResNetMod(Model):
#     def __init__(self, config):
#         super().__init__(config)
#         if config['no_norm']:
#             self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
#         else:
#             self.backbone = ResNet18(num_classes=config['num_classes'])
#         self.prototype = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False)
#         self.backbone.linear = None

#     def forward(self, x):
#         # Convolution layers
#         feature_embedding = self.backbone(x)
#         logits = self.prototype(feature_embedding)
#         return logits

#     def get_embedding(self, x):
#         feature_embedding = self.backbone(x)
#         logits = self.prototype(feature_embedding)
#         return feature_embedding, logits


# class ResNetModNH(Model):
#     def __init__(self, config):
#         super().__init__(config)
#         self.return_embedding = config['FedNH_return_embedding']
#         if config['no_norm']:
#             self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
#         else:
#             self.backbone = ResNet18(num_classes=config['num_classes'])
#         temp = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False).state_dict()['weight']
#         self.prototype = nn.Parameter(temp)
#         self.backbone.linear = None
#         self.scaling = torch.nn.Parameter(torch.tensor([20.0]))
#         self.activation = None

#     def forward(self, x):
#         feature_embedding = self.backbone(x)
#         feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
#         if self.prototype.requires_grad == False:
#             normalized_prototype = self.prototype
#         else:
#             prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#             normalized_prototype = torch.div(self.prototype, prototype_norm)
#         logits = torch.matmul(feature_embedding, normalized_prototype.T)
#         logits = self.scaling * logits
#         self.activation = self.backbone.activation
#         if self.return_embedding:
#             return feature_embedding, logits
#         else:
#             return logits

class tumorModel_runCifarDataset_FedNH(Model):
    def __init__(self, config):
        super().__init__(config)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.prototype = nn.Linear(512, config['num_classes'])
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return x, logits


class tumorModel(Model):
    def __init__(self,config):
        super().__init__(config)
        
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=0) # kernel_siae 4->3
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(6*6*128,512)
        self.prototype = nn.Linear(512,config['num_classes'])
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
        
        
        
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return x, logits


class tumorModelNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.linear1 = nn.Linear(6*6*128, 512)
        self.linear2 = nn.Linear(512, 192)
        
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # print(f"Input size: {x.size()}")
        x = self.relu(self.bn1(self.conv1(x)))
        # print(f"After Conv1 and BN1: {x.size()}")
        x = self.pool(x)
        # print(f"After Pool1: {x.size()}")
        x = self.relu(self.bn2(self.conv2(x)))
        # print(f"After Conv2 and BN2: {x.size()}")
        x = self.pool(x)
        # print(f"After Pool2: {x.size()}")
        x = self.relu(self.bn3(self.conv3(x)))
        # print(f"After Conv3 and BN3: {x.size()}")
        x = self.pool2(x)
        # print(f"After Pool2_2: {x.size()}")
        x = self.relu(self.bn4(self.conv4(x)))
        # print(f"After Conv4 and BN4: {x.size()}")
        x = self.flatten(x)
        # print(f"After Flatten: {x.size()}")
        x = self.relu(self.linear1(x))
        # print(f"After FC1: {x.size()}")
        feature_embedding = self.dropout(x)
        feature_embedding = F.relu(self.linear2(feature_embedding))
        # print(f"After Linear2: {feature_embedding.size()}")
        
        # Normalize feature embedding
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        # print(f"After Normalizing feature embedding: {feature_embedding.size()}")

        # Normalize prototype
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        # print(f"Normalized Prototype size: {normalized_prototype.size()}")

        # Calculate logits
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits
        # print(f"Logits size: {logits.size()}")
        
        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


# class tumorModel_cifar10_dataNH(Model):
#     def __init__(self, config):
#         super().__init__(config)
#         self.return_embedding = config['FedNH_return_embedding']
        
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
        
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
        
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
        
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
#         self.linear1 = nn.Linear(6*6*128, 512)
#         self.linear2 = nn.Linear(512, 192)
        
#         temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
#         self.prototype = nn.Parameter(temp)
#         self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
#         self.flatten = nn.Flatten()
#         self.relu = nn.ReLU() 
#         self.dropout = nn.Dropout(0.5)
        
#     def forward(self, x):
#         print(x.size())
#         x = self.relu(self.bn1(self.conv1(x)))
#         print(x.size())
#         x = self.pool(x)
#         print(x.size())
#         x = self.relu(self.bn2(self.conv2(x)))
#         print(x.size())
#         x = self.pool(x)
#         print(x.size())
#         x = self.relu(self.bn3(self.conv3(x)))
#         print(x.size())
#         x = self.pool2(x)
#         print(x.size())
#         x = self.relu(self.bn4(self.conv4(x)))
#         print(x.size())
#         x = self.flatten(x)
#         print(x.size())
#         x = self.relu(self.linear1(x))
#         print(x.size())
#         feature_embedding = self.dropout(x)
#         print(x.size())
#         feature_embedding = F.relu(self.linear2(feature_embedding))
#         print(feature_embedding.size())
        
#         # Normalize feature embedding
#         feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#         feature_embedding = torch.div(feature_embedding, feature_embedding_norm)

#         # Normalize prototype
#         if not self.prototype.requires_grad:
#             normalized_prototype = self.prototype
#         else:
#             prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#             normalized_prototype = torch.div(self.prototype, prototype_norm)

#         # Calculate logits
#         logits = torch.matmul(feature_embedding, normalized_prototype.T)
#         logits = self.scaling * logits
        
#         if self.return_embedding:
#             return feature_embedding, logits
#         else:
#             return logits

class tumorModel_cifar10_dataNH(Model):
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Corrected input size for self.linear1
        self.linear1 = nn.Linear(3 * 3 * 128, 512)
        self.linear2 = nn.Linear(512, 192)
        
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # print(x.size())
        x = self.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = self.relu(self.bn2(self.conv2(x)))
        # print(x.size())
        x = self.pool(x)
        # print(x.size())
        x = self.relu(self.bn3(self.conv3(x)))
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())
        x = self.relu(self.bn4(self.conv4(x)))
        # print(x.size())
        x = self.flatten(x)
        # print(x.size())
        x = self.relu(self.linear1(x))
        # print(x.size())
        feature_embedding = self.dropout(x)
        # print(x.size())
        feature_embedding = F.relu(self.linear2(feature_embedding))
        # print(feature_embedding.size())
        
        # Normalize feature embedding
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)

        # Normalize prototype
        if not self.prototype.requires_grad:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)

        # Calculate logits
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits
        
        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits

class tumorModel_cifar10_dataProto(Model):
    def __init__(self,config):
        super().__init__(config)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(4*4*128, 512)  # Adjusted for CIFAR-10 image size
        self.prototype = nn.Linear(512, config['num_classes'], bias=False)
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5)
        
    def get_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return x, logits
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.prototype(x)
        return logits
