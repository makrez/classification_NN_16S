import torch
import torch.nn as nn

class SmallModel(nn.Module):
    def __init__(self, input_length, num_classes):
        super(SmallModel, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 8, kernel_size=10, stride=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(6664, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # Adjust the weight initialization based on the adjusted flattened size
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        data = x[:, :6, :]
        for i, layer in enumerate(self.features):
            data = layer(data)
        data = data.view(data.size(0), -1)
        for i, layer in enumerate(self.classifier):
            data = layer(data)
        return data



class ConvClassifier2(nn.Module):
    def __init__(self, input_length, num_classes):
        super(ConvClassifier2, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(12, 8, kernel_size=10, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 4, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )
            # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(4*1559, 1200),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(1200, 600), # Match output size of previous layer
            nn.ReLU(inplace=True),
            nn.Linear(600, 281), # Match output size of previous layer
            nn.ReLU(inplace=True),
            nn.Linear(281, num_classes)
)


        # Initialize weights with He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        data = x[:, :6, :]
        for i, layer in enumerate(self.features):
            data = layer(data)
        data = data.view(data.size(0), -1)
        for i, layer in enumerate(self.classifier):
            data = layer(data)
        return data
    
class ModelWithDropout(nn.Module):
    def __init__(self, input_length, num_classes):
        super(ModelWithDropout, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(12, 8, kernel_size=10, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 4, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )

        # Calculate the expected size of the input to the linear layer
        input_size = 4 * 1559  # Adjusted input size based on the flattened shape

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1200, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 281),
            nn.ReLU(inplace=True),
            nn.Linear(281, num_classes)
        )

    def forward(self, x):
        data = x[:, :6, :]
        for i, layer in enumerate(self.features):
            data = layer(data)
        data = data.view(data.size(0), -1)
        for i, layer in enumerate(self.classifier):
            data = layer(data)
        return data
    
class LargerModel(nn.Module):
    def __init__(self, input_length, num_classes):
        super(LargerModel, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 16, kernel_size=10, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 8, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(12472, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        data = x[:, :6, :]
        for i, layer in enumerate(self.features):
            data = layer(data)
        data = data.view(data.size(0), -1)
        for i, layer in enumerate(self.classifier):
            data = layer(data)
        return data
