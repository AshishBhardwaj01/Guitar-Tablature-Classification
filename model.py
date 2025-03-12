# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GuitarTabNet(nn.Module):
#     def __init__(self, input_shape, num_frets=19):
#         super(GuitarTabNet, self).__init__()

#         # Convolutional Layers
#         self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Calculate flattened size after convolutions
#         with torch.no_grad():
#             self.flatten_size = self._get_flatten_size(input_shape)

#         # Fully Connected Branches for each guitar string
#         self.branches = nn.ModuleList([self._create_branch(self.flatten_size, num_frets) for _ in range(6)])

#     def _get_flatten_size(self, input_shape):
#         dummy_input = torch.zeros(1, *input_shape)
#         x = self.pool(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))))
#         return x.view(1, -1).size(1)

#     def _create_branch(self, input_dim, num_frets):
#         return nn.Sequential(
#             nn.Linear(input_dim, 152),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(152, 76),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(76, num_frets)
#         )

#     def forward(self, x):
#         # Shared CNN feature extraction
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)

#         # Flatten for fully connected layers
#         x = torch.flatten(x, start_dim=1)

#         # Forward pass for each guitar string
#         outputs = [F.log_softmax(branch(x), dim=1) for branch in self.branches]

#         return outputs


# def get_model(input_shape, learning_rate=0.01, momentum=0.8, epochs=30):
#     model = GuitarTabNet(input_shape)

#     # Optimizer with SGD and learning rate decay
#     decay = learning_rate / epochs
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

#     return model, optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GuitarTabNet(nn.Module):
#     def __init__(self, input_shape, num_frets=19):
#         super(GuitarTabNet, self).__init__()

#         # Load Pretrained ResNet18 and modify first conv layer to accept spectrogram input
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1, bias=False)
#         self.resnet.fc = nn.Linear(512, 256)  # Replace Identity layer with FC layer

#         # Fully Connected Layers for Each String
#         self.branches = nn.ModuleList([self._create_branch(256, num_frets) for _ in range(6)])

#     def _create_branch(self, input_dim, num_frets):
#         return nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#             nn.Dropout(0.2),
#             nn.Linear(64, num_frets)
#         )

#     def forward(self, x):
#         x = self.resnet(x)  # Feature extraction
#         outputs = [F.log_softmax(branch(x), dim=1) for branch in self.branches]
#         return outputs


# def get_model(input_shape, learning_rate=0.001):
#     model = GuitarTabNet(input_shape)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#     return model, optimizer
def __init__(self, input_channels=3, num_frets=19, dropout_rate=0.3):
        super(ImprovedGuitarTabModel, self).__init__()
        
        # Load Pretrained ResNet18 and modify first conv layer to accept RGB/spectrogram input
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.resnet.fc = nn.Linear(512, 256)  # Replace final layer with FC

        # Fully Connected Layers for Each String
        self.branches = nn.ModuleList([self._create_branch(256, num_frets, dropout_rate) for _ in range(6)])

    def _create_branch(self, input_dim, num_frets, dropout_rate):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_frets)
        )

    def forward(self, x):
        x = self.resnet(x)  # Feature extraction
        outputs = [F.log_softmax(branch(x), dim=1) for branch in self.branches]
        return outputs


def get_model(input_shape, learning_rate=0.001):
    model = ImprovedGuitarTabModel(input_channels=input_shape[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return model, optimizer
