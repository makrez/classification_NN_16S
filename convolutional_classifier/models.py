import torch
import torch.nn as nn

class ConvClassifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super(ConvClassifier, self).__init__()
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
            nn.MaxPool1d(2, stride=2)
        )

        # Define the classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(562 * 12, 562*6),
            nn.ReLU(inplace=True),
            nn.Linear(562 * 6, 562*3),
            nn.ReLU(inplace=True),
            nn.Linear(562 * 3, 562),
            nn.ReLU(inplace=True),
            nn.Linear(562, 281),
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
