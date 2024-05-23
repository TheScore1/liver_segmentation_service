import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from monai.networks.nets import UNet
import time

# Загрузка сохраненной модели
def load_model(model_path, device):
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Функция для предсказания маски
def predict_mask(image, model, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output).cpu().numpy().squeeze()
    return pred

# Отображение изображений и масок
def show_image_and_mask(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Predicted Mask")
    st.pyplot(fig)

# Настройки Streamlit
st.title("Liver Segmentation using UNet")
st.write("Upload an image to get the liver segmentation mask.")

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "liver_segmentation_model.pth"
model = load_model(model_path, device)

# Загрузка изображения
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_placeholder = st.empty()
    text_placeholder = st.empty()
    
    image_placeholder.image(image, caption='Uploaded Image', use_column_width=True)
    text_placeholder.write("Classifying...")
    
    # Предсказание маски
    mask = predict_mask(image, model, device)
    
    # Задержка перед отображением результата
    time.sleep(2)

    # Очистка места для нового контента
    image_placeholder.empty()
    text_placeholder.empty()

    # Отображение результата
    show_image_and_mask(np.array(image), mask)

#streamlit run app.py
