import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from monai.networks.nets import UNet
import nibabel as nib
import time
import tempfile
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    if image.mode == 'RGB':
        image = image.convert('L')

    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    
    transformed = transform(image=np.array(image))
    image = transformed['image'].unsqueeze(0).to(device)

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

# Функция для загрузки и обработки NIfTI файлов
def load_nifti_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        nifti_img = nib.load(tmp_path)
        nifti_data = nifti_img.get_fdata()
        return nifti_data
    except Exception as e:
        st.error(f"Error loading NIfTI file: {e}")
        return None

def normalize_nifti_image(nifti_slice):
    nifti_slice = (nifti_slice - nifti_slice.min()) / (nifti_slice.max() - nifti_slice.min())
    nifti_slice = (nifti_slice * 255).astype(np.uint8)
    return nifti_slice


# Настройки Streamlit
st.title("Liver Segmentation using UNet")
st.write("Upload an image to get the liver segmentation mask.")

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "e-18-liver_segmentation_model.pth"
model = load_model(model_path, device)

# Загрузка изображения
uploaded_file = st.file_uploader("Choose an image (jpg or nii)...", type=["jpg", "png", "nii"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".nii"):
        # Обработка NIfTI файла
        nifti_data = load_nifti_file(uploaded_file)
        if nifti_data is not None:
            num_slices = nifti_data.shape[2]
            slice_idx = st.slider("Select Slice", 0, num_slices - 1, num_slices // 2)
            selected_slice = nifti_data[:, :, slice_idx]
            normalized_slice = normalize_nifti_image(selected_slice)
            image = Image.fromarray(normalized_slice, 'L')
            ##image = Image.fromarray(np.uint8((selected_slice - selected_slice.min()) * 255 / (selected_slice.max() - selected_slice.min())), 'L')
        else:
            st.error("Failed to load NIfTI file.")
    else:
        # Обработка jpg файла
        image = Image.open(uploaded_file)
        selected_slice = np.array(image)

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
    show_image_and_mask(selected_slice, mask)
