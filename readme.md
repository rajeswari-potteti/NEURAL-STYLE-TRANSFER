COMPANY : CODTECH IT SOLUTIONS

NAME : POTTETI RAJESWARI

INTERN ID : CT06DM1193

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 6 WEEKS

MENTOR : NEELA SANTHOSH

NEURAL-STYLE-TRANSFER
About the Project
This is a Python-based Neural Style Transfer tool built using PyTorch and a pre-trained VGG19 convolutional neural network. The tool blends the content of one image with the style of another to produce a visually artistic image, preserving key features from the content and infusing the texture and color characteristics of the style.

The main goal is to generate a stylized version of a photograph, similar to how a famous painter might reimagine a real-world image.

What This Project Does
The project takes two input images:

A content image (the base photograph whose structure you want to preserve).
A style image (an artwork or texture you want to apply).
Using a pre-trained VGG19 model, the script extracts content and style features and optimizes a copy of the content image such that:

Content loss (difference from original structure) is minimized.
Style loss (difference in textures and patterns) is minimized.
The result is a hybrid image that preserves the structure of the content image but reflects the artistic texture of the style image.

After completing the transfer, the stylized image is saved locally and displayed.

Why This Was Built
The project was built to apply deep learning techniques for artistic purposes. While most machine learning applications focus on analysis and prediction, this task uses models creatively, demonstrating how neural networks can be used to generate art and perform complex image transformations.

It also serves as a practical exercise in working with:

Pre-trained CNNs (VGG19)
Feature extraction
Loss functions and optimization
Image preprocessing and postprocessing
This tool is ideal for understanding how convolutional layers learn and represent visual hierarchies in image data.

Prerequisites
Ensure you have Python 3.7 or higher installed.

Install required Python libraries via pip: pip install -r requirements.txt

Also ensure you have your content and style images (e.g., content.jpg and style.jpg) in the same folder as the script.

How to Use
Place your content and style images in the project directory.

Rename them (or update the paths in the script) as:

content.jpg

style.jpg

Run the script:

python neural_style_transfer.py

The output will be:

Displayed in a window titled "Stylized Output"

Saved as stylized_output.jpg in the current directory

output
Content Image	Style Image	Stylized Image
		
