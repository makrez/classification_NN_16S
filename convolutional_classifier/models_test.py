# import torch
# import torch.nn as nn

# class ConvClassifier(nn.Module):
#     def __init__(self, input_length, num_classes):
#         super(ConvClassifier, self).__init__()
#         self.input_length = input_length
#         self.num_classes = num_classes

#         self.features = nn.Sequential(
#             nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2, stride=2),
#             nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2, stride=2),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2, stride=2),
#         )

#         # Calculate the size of the final feature map after convolutional and pooling layers
#         feature_map_size = input_length // (2 * 2 * 2)

#         # Define the classifier layers
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * feature_map_size, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, num_classes),
#         )


#     def forward(self, x):
#         print("Input tensor shape:", x.shape)
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         print("Output tensor shape before classifier:", x.shape)
#         x = self.classifier(x)
#         print("Output tensor shape:", x.shape)
#         return x

import torch
import torch.nn as nn

conv_layer1 = nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1)
relu_activation1 = nn.ReLU(inplace=True)
conv_layer2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
relu_activation2 = nn.ReLU(inplace=True)
pool2 = nn.MaxPool1d(2, stride=2)
conv_layer3 = nn.Conv1d(32, 12, kernel_size=3, stride=1, padding=1)
relu_activation3 = nn.ReLU(inplace=True)
pool3 = nn.MaxPool1d(2, stride=2)

input_length = 4500
dummy_input = torch.randn(1, 6, 4500)  # Example input shape: (batch_size, channels, input_length)
print(dummy_input)
feature_map_size = input_length // (2 * 2)
print(f"feature_map size: {feature_map_size}")


output = conv_layer1(dummy_input)
print("Output shape after conv_layer1:", output.shape)

output = relu_activation1(output)
print("Output shape after relu_activation1:", output.shape)

output = conv_layer2(output)
print("Output shape after conv_layer2:", output.shape)

output = relu_activation2(output)
print("Output shape after relu_activation2:", output.shape)

output = pool2(output)
print("Output shape after pool2:", output.shape)

output = conv_layer3(output)
print("Output shape after conv_layer3:", output.shape)

output = relu_activation3(output)
print("Output shape after relu_activation3:", output.shape)

output = pool3(output)
print("Output shape after pool3:", output.shape)

