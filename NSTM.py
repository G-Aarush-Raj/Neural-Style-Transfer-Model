import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_image(path, max_size=256):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)

class VGGFeatures(nn.Module):
    def _init_(self):
        super(VGGFeatures, self)._init_()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.layers = {
            '0': 'conv1_1', '5': 'conv2_1',
            '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2'
        }
        self.vgg = vgg

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

content_img = load_image("content.jpg", 256)
style_img = load_image("style.jpg", 256)
generated_img = content_img.clone().requires_grad_(True)
model = VGGFeatures().to(device)

alpha = 1e2
beta = 1e6
optimizer = optim.Adam([generated_img], lr=0.003)
content_losses, style_losses, total_losses = [], [], []

with torch.no_grad():
    content_features = model(content_img)
    style_features = model(style_img)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

for step in range(1, 2001):
    generated_features = model(generated_img)
    content_loss = torch.mean((generated_features['conv4_2'] - content_features['conv4_2']) ** 2)
    style_loss = 0
    for layer in style_features:
        gen_feature = generated_features[layer]
        gen_gram = gram_matrix(gen_feature)
        style_gram = style_grams[layer]
        style_loss += torch.mean((gen_gram - style_gram) ** 2)
    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    content_losses.append(content_loss.item())
    style_losses.append(style_loss.item())
    total_losses.append(total_loss.item())
    if step % 100 == 0:
        print(f"Step {step:3d} | Total: {total_loss.item():.4f} | Content: {content_loss.item():.4f} | Style: {style_loss.item():.4f}")

final_img = tensor_to_image(generated_img)
final_np = np.array(final_img)
content_np = np.array(tensor_to_image(content_img))

mse_value = np.mean((final_np - content_np) ** 2)
ssim_value = ssim(content_np, final_np, channel_axis=2)
psnr_value = psnr(content_np, final_np)

print("\nEvaluation Metrics:")
print(f"MSE: {mse_value:.4f}")
print(f"SSIM: {ssim_value:.4f}")
print(f"PSNR: {psnr_value:.2f} dB")
print(f"Final Content Loss: {content_losses[-1]:.6f}")
print(f"Final Style Loss: {style_losses[-1]:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(content_losses, label='Content Loss', color='blue')
plt.plot(style_losses, label='Style Loss', color='red')
plt.plot(total_losses, label='Total Loss', color='green')
plt.title("Loss vs Iterations (Neural Style Transfer)")
plt.xlabel("Iteration")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(np.array(final_img))
plt.title("Final Stylized Image Output")
plt.axis('off')
plt.show()