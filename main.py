import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from rembg import remove

# -----------------------
# Setup
# -----------------------
# TODO: Set random seed and device
torch.manual_seed(40)
# TODO: Define transforms

# TODO: Load dataset
img_route = r"C:\Users\ivanl\OneDrive\Desktop\Singapore_Battery_Dataset M"
Alkaline = "Alkaline"
Lithium_Ion = "Lithium Ion (338 imgs)"
Nickel_Cadmium = "Nickel Cadmium (263 imgs)"
Nickel_MetalHydride = "Nickel Metal Hydride (439 imgs)"
Validation = "validation"

# TODO: Create data loaders
def load_img_dir(img_root_path, class_path, class_num):
    full_image_path = os.path.join(img_root_path, class_path)
    img_list = []
    mask_list = []
    
    for image_name in tqdm(os.listdir(full_image_path)):
        fname = os.path.join(full_image_path, image_name)
        try:
            img = load_resize_to_pil(fname)
        except:
            continue
        
        new_img = make_square(img, min_size=256, fill_color=(256, 256, 256, 256))
        rembg_img = remove(new_img)
        rembg_img = np.array(rembg_img).mean(axis=2)
        rembg_img = rembg_img > 0
        rembg_img = rembg_img * 1.0
        mask = np.zeros((256, 256, 4))
        mask[:,:,class_num] += rembg_img
        mask = np.transpose(mask, axes=[2,1,0])
        mask = torch.from_numpy(mask).to(torch.int64)
        mask_list.append(mask)
        
        img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(128)])
        img_tensor = img_transforms(new_img)
        img_list.append(img_tensor)
        
    classes = [class_num] * len(img_list)
    
    return img_list, classes, mask_list


def load_resize_to_pil(path, max_size=(256, 256)):
    cv_img = cv2.imread(path)
    if cv_img is None:
        raise ValueError(f"Could not load image: {path}")

    # BGR -> RGB
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # Current size
    h, w = cv_img.shape[:2]
    
    # Compute scale (preserve aspect ratio)
    scale = min(max_size[0] / w, max_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize only if needed
    if scale < 1.0:
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return Image.fromarray(cv_img)


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im.convert('RGB')

images = []
classes = []
masks_list = []

image, cs, masks = load_img_dir(img_route, Alkaline, 0)
images.extend(image)
classes.extend(cs)
masks_list.extend(masks)

image, cs, masks = load_img_dir(img_route, Lithium_Ion, 1)
images.extend(image)
classes.extend(cs)
masks_list.extend(masks)

image, cs, masks = load_img_dir(img_route, Nickel_Cadmium, 2)
images.extend(image)
classes.extend(cs)
masks_list.extend(masks)

image, cs, masks = load_img_dir(img_route, Nickel_MetalHydride, 3)
images.extend(image)
classes.extend(cs)
masks_list.extend(masks)

images = torch.stack(images)
classes = torch.LongTensor(classes)
masks = torch.stack(masks_list)

idxs = torch.linspace(0, len(images)-1, len(images), dtype=torch.int32)
idxs = idxs[torch.randperm(len(images))]
train_idxs = idxs[:int(len(images)*0.9)]
test_idxs = idxs[int(len(images)*0.9):]

train_data = TensorDataset(images[train_idxs], classes[train_idxs], masks[train_idxs])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(images[test_idxs], classes[test_idxs], masks[test_idxs])
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# -----------------------
# Model
# -----------------------
# TODO: Define custom model class
class MobileNetV3Segmentation(nn.Module):
    """Segmentation model using MobileNetV3 encoder with decoder"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # TODO: Load pretrained MobileNetV3 and extract encoder
        mobilenet = mobilenet_v3_small()
        self.encoder = mobilenet
        # TODO: Build decoder network to upsample back to input size
        encoder_output = 576
        
        self.conv1 = nn.ConvTranspose2d(encoder_output, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        
        self.conv6 = nn.Conv2d(32, 4, kernel_size=1)
        # Hint: Encoder outputs 7x7 feature maps, need to reach 224x224
        # Hint: Use transposed convolutions for upsampling
    
    def forward(self, x):
        # TODO: Pass input through encoder then decoder
        # TODO: Return single-channel output (binary mask)
        x = self.encoder(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x
    
# TODO: Instantiate model
model = MobileNetV3Segmentation()
# TODO: Move model to device
model.cpu()
# TODO: Define loss function
criterion = nn.CrossEntropyLoss()
# TODO: Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------
# Training Function
# -----------------------
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    accuracies = []
    losses = []
    
    for epoch in range(num_epochs):
        num_correct = 0
        model.train()
        
        for imgs, labels, masks in tqdm(train_loader):
            imgs, labels, masks = imgs.cpu(), labels.cpu(), masks.cpu()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            num_correct += (outputs.argmax(dim=1) == masks).detach().numpy().sum()
            
        print(f"Epoch {epoch+1} complete")
        accuracies.append(num_correct / len(images))
        print(num_correct / len(images))

def eval_model(model, test_loader, criterion):
    # TODO: Implement evaluation function
    accuracies = []
    losses = []
    num_correct = 0
    
    model.eval()
    
    for imgs, labels, masks in tqdm(test_loader):
        imgs, labels, masks = imgs.cpu(), labels.cpu(), masks.cpu()
        outputs = model(imgs)
        num_correct += (outputs.argmax(dim=1) == masks).detach().numpy().sum()
        
    print(num_correct / len(images))

# -----------------------
# Run Training
# -----------------------
# TODO: Call training function
train_model(model, train_loader, criterion, optimizer)
eval_model(model, test_loader, criterion)

def plot_examples(model, images, labels):
    predictions = model(images).argmax(dim=1)
    fig, axs = plt.subplots(4, 8, figsize=(10,10))
    axs = axs.flatten()
    for ax, img, label, prediction in zip(axs, images, labels, predictions):
        transform_image = transforms.ToPILImage()(img)
        ax.imshow(transform_image)
        ax.set_title(f"{prediction.item()}, {label.item()}")
    fig.tight_layout()
    fig.show()

example_imgs, example_classes = next(iter(train_loader))
# plot_examples(model, example_imgs, example_classes)
# TODO: Call evaluation function

# -----------------------
# Saving & Loading Models
# -----------------------
def save_model(model, filepath):
    # TODO: Save model
    pass

def load_model(filepath, num_classes):
    # TODO: Load model
    pass
