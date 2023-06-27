import torch
import torch.nn as nn

class ConvClassifier(nn.Module):
    def __init__(self, input_length, num_classes):
        super(ConvClassifier, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(6840, 1200),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(1200, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 281),
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
        # The data is in the first 5 channels and the mask is in the 6th channel
        data = x[:, :5, :]
        mask = x[:, 5, :]

        # Multiply data by mask (this sets the masked values to zero)
        data = data * mask.unsqueeze(1)  # we need to add an extra dimension to mask for broadcasting to work

        data = self.features(data)

        data = data.view(data.size(0), -1)
        data = self.classifier(data)
        return data

class ConvClassifier2(nn.Module):
    def __init__(self, input_length, num_classes):
        super(ConvClassifier2, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(75000, 1200),
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
        # The data is in the first 5 channels and the mask is in the 6th channel
        data = x[:, :5, :]
        mask = x[:, 5, :]

        # Multiply data by mask (this sets the masked values to zero)
        data = data * mask.unsqueeze(1)  # we need to add an extra dimension to mask for broadcasting to work
        
        for i, layer in enumerate(self.features):
            data = layer(data)
            #print(f"After layer {i+1}, shape is {data.shape}")

        data = data.view(data.size(0), -1)
        #print(f"Shape before classifier: {data.shape}")
        
        data = self.classifier(data)
        return data


class ConvClassifierBacillus(nn.Module):
    def __init__(self, input_length, num_classes):
        super(ConvClassifierBacillus, self).__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )
        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(6840, 1200),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(1200, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 281),
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
        # The data is in the first 5 channels and the mask is in the 6th channel
        data = x[:, :5, :]
        mask = x[:, 5, :]

        # Multiply data by mask (this sets the masked values to zero)
        data = data * mask.unsqueeze(1)  # we need to add an extra dimension to mask for broadcasting to work

        data = self.features(data)

        data = data.view(data.size(0), -1)
        data = self.classifier(data)
        return data