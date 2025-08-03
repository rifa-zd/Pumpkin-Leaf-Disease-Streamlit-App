import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import streamlit as st

# Custom CNN Definition

class CustomCNN(nn.Module): 

    def __init__(self, num_classes): 
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 28 * 28, 1024), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
    


# Model Loading Helpers

def load_custom_cnn (path, num_classes, device):
    model = CustomCNN(num_classes)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

def load_resnet50(path, num_classes, device):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model



# Streamlit APP Setup

st.title("Pumpkin Leaf Disease Classification App")

model_choice = st.sidebar.selectbox("Select a Model", ["Custom CNN", "ResNet50"])

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

CLASS_NAMES = ["Bacterial Spot", "Downy Mildew", "Mosaic", "Healthy", "Powdery Mildew"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
  cnn = load_custom_cnn("custom_cnn_model.pth", NUM_CLASSES, DEVICE)
  resnet = load_resnet50("transfer_learning_resnet50.pth", NUM_CLASSES, DEVICE)
  return cnn, resnet

custom_cnn, resnet50 = load_model()



# Image Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                         std=[0.5, 0.5, 0.5])
])

# Prediction & Display

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model = custom_cnn if model_choice == "Custom CNN" else resnet50

    with torch.no_grad():
        pred = model(tensor)
        probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
    
    top_idx = np.argsort(probs)[::-1][:3]
    st.subheader("Top 3 Predictions:")
    for idx in top_idx:
        st.write(f"{CLASS_NAMES[idx]}: {probs[idx] * 100:.2f}%")




