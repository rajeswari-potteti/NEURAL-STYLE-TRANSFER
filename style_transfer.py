"""
Neural Style Transfer 

Description:
This script performs neural style transfer to blend the style of one image
(style image) with the content of another (content image). It uses a pre-trained
VGG19 model to extract style and content features and optimizes the input image
to generate a stylized output.

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256

# Image Preprocessing Transforms
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def image_loader(image_path):
    """
    Loads and preprocesses an image from the given path.
    """
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)  # Add batch dimension
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    """
    Displays a tensor as an image using matplotlib.
    """
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Define the Content Loss Module
class ContentLoss(nn.Module):
    """
    Computes the content loss between the target and input feature maps.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Define the Gram Matrix for Style Loss
def gram_matrix(input):
    """
    Computes the Gram matrix for the given input tensor.
    """
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# Define the Style Loss Module
class StyleLoss(nn.Module):
    """
    Computes the style loss between the Gram matrices of input and target.
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Define the Normalization Layer
class Normalization(nn.Module):
    """
    Applies image normalization using ImageNet mean and std.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Build the Model with Style and Content Loss Layers
def get_style_model_and_losses(cnn, mean, std, style_img, content_img):
    """
    Builds the model by inserting style and content loss layers.
    Returns the model and lists of the loss modules.
    """
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(mean, std).to(device)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0  # increment for conv layers

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Truncate model after last loss layer
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j+1]

    return model, style_losses, content_losses

# Run the Style Transfer
def run_style_transfer(cnn, mean, std, content_img, style_img, input_img,
                       num_steps=300, style_weight=1e6, content_weight=1):
    """
    Executes the style transfer optimization.
    """
    print("Running style transfer...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, mean, std, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]} - Style Loss: {style_score:.4f}, Content Loss: {content_score:.4f}")
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# Load VGG19 Model
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Load content and style images
content_path = "content.jpg"  # Replace if needed
style_path = "style.jpg"      # Replace if needed

content_img = image_loader(content_path)
style_img = image_loader(style_path)
input_img = content_img.clone()

assert content_img.size() == style_img.size(), "Content and style images must be the same size."

# Run the transfer
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

# Display and save result
imshow(output, title="Stylized Output")
save_image(output, "stylized_output.jpg")
print("Stylized image saved as stylized_output.jpg")
