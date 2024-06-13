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
from sklearn.metrics import jaccard_score

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
def show_image_and_masks(image, original_mask, predicted_mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[1].imshow(original_mask, cmap='gray')
    axes[1].set_title("Original Mask")
    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title("Predicted Mask")
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

# Функция для подсчета индекса жаккара
def calculate_jaccard_index(original_mask, predicted_mask):
    original_mask_flat = original_mask.flatten()
    predicted_mask_flat = (predicted_mask > 0.5).astype(np.uint8).flatten()
    return jaccard_score(original_mask_flat, predicted_mask_flat, average='binary')

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
    show_image_and_masks(selected_slice, np.zeros_like(mask), mask)

# Загрузка изображения и маски для расчета индекса Жаккара
st.write("Upload an image and the corresponding ground truth mask to calculate the Jaccard index.")

uploaded_image_file = st.file_uploader("Choose an image for Jaccard calculation (jpg or nii)...", type=["jpg", "png", "nii"], key="image_file")
uploaded_mask_file = st.file_uploader("Choose the corresponding mask (jpg or nii)...", type=["jpg", "png", "nii"], key="mask_file")

if uploaded_image_file is not None and uploaded_mask_file is not None:
    if uploaded_image_file.name.endswith(".nii") and uploaded_mask_file.name.endswith(".nii"):
        # Обработка NIfTI файла
        nifti_image_data = load_nifti_file(uploaded_image_file)
        nifti_mask_data = load_nifti_file(uploaded_mask_file)
        if nifti_image_data is not None and nifti_mask_data is not None:
            num_slices = nifti_image_data.shape[2]
            slice_idx = st.slider("Select Slice for Jaccard", 0, num_slices - 1, num_slices // 2, key="jaccard_slider")
            selected_image_slice = nifti_image_data[:, :, slice_idx]
            selected_mask_slice = nifti_mask_data[:, :, slice_idx]
            normalized_image_slice = normalize_nifti_image(selected_image_slice)
            normalized_mask_slice = normalize_nifti_image(selected_mask_slice)
            image = Image.fromarray(normalized_image_slice, 'L')
            original_mask = normalized_mask_slice
        else:
            st.error("Failed to load NIfTI files.")
    else:
        # Обработка jpg файлов
        image = Image.open(uploaded_image_file)
        original_mask = Image.open(uploaded_mask_file).convert('L')
        selected_image_slice = np.array(image)
        original_mask = np.array(original_mask)

    # Предсказание маски
    predicted_mask = predict_mask(image, model, device)

    # Ensure masks are binary
    original_mask_binary = (original_mask > 127).astype(np.uint8)
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

    # Расчет индекса Жаккара
    jaccard_index = calculate_jaccard_index(original_mask_binary, predicted_mask_binary)
    st.write(f"Jaccard Index: {jaccard_index:.4f}")

    # Отображение результата
    show_image_and_masks(selected_image_slice, original_mask_binary, predicted_mask_binary)
