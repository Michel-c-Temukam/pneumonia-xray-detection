import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Détection de pneumonie", page_icon="🫁")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ["NORMAL", "PNEUMONIA"]

@st.cache_resource
def load_model():
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load("best_model_finetuned.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
target_layers = [model.features.norm5]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

st.title("🫁 Détection de pneumonie sur radiographie thoracique")
st.caption("Projet académique — DenseNet121 fine-tuné. ⚠️ Outil pédagogique, ne constitue PAS un diagnostic médical.")

uploaded_file = st.file_uploader("Uploade une radiographie thoracique", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)
    img_tensor.requires_grad_(True)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred_idx = probs.argmax().item()
    st.subheader(f"Prédiction : {CLASSES[pred_idx]}")
    st.write(f"NORMAL : {probs[0]:.2%} | PNEUMONIA : {probs[1]:.2%}")

    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img_np = np.clip(img_np, 0, 1)
    heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    st.image(heatmap, caption="Zone d'attention du modèle (Grad-CAM)", use_container_width=True)
  
