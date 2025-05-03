import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
import torchvision.transforms.functional as TF

st.set_page_config(
    page_title="Editor de Fotos IA",
    layout="centered",
    page_icon=":camera:",
    initial_sidebar_state="expanded"
)

# Estilo customizado (tema preto e laranja)
st.markdown("""
    <style>
    body {
        background-color: #111;
        color: #FFA500;
    }
    .css-1v0mbdj, .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2 {
        color: #FFA500 !important;
    }
    .stSlider > div > div {
        background: #FFA500;
    }
    .stButton>button {
        background-color: #FFA500;
        color: black;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Editor de Fotos Inteligente")
st.write("Edição automática com IA e estilo CapCut")

uploaded_file = st.file_uploader("Envie sua imagem", type=["jpg", "jpeg", "png", "webp"])

@st.cache_resource

def load_model():
    model = mobilenet_v2(pretrained=True)
    model.eval()
    return model

@torch.no_grad()
def aplicar_ia_classificacao(image):
    model = load_model()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(image).unsqueeze(0)
    output = model(tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem Original", use_column_width=True)

        st.sidebar.header("Efeitos automáticos")
        auto_enhance = st.sidebar.checkbox("Melhoria automática de brilho, contraste e nitidez", value=True)
        aplicar_filtro = st.sidebar.selectbox("Filtro artístico", ["Nenhum", "Contorno", "Detalhe", "Bordas", "Desfoque leve"])
        aplicar_ia = st.sidebar.checkbox("Usar IA para entender o conteúdo da imagem", value=True)

        if auto_enhance:
            image = ImageEnhance.Brightness(image).enhance(1.2)
            image = ImageEnhance.Contrast(image).enhance(1.2)
            image = ImageEnhance.Sharpness(image).enhance(1.3)

        if aplicar_filtro == "Contorno":
            image = image.filter(ImageFilter.CONTOUR)
        elif aplicar_filtro == "Detalhe":
            image = image.filter(ImageFilter.DETAIL)
        elif aplicar_filtro == "Bordas":
            image = image.filter(ImageFilter.FIND_EDGES)
        elif aplicar_filtro == "Desfoque leve":
            image = image.filter(ImageFilter.GaussianBlur(1.5))

        if aplicar_ia:
            predicted_class = aplicar_ia_classificacao(image)
            st.success(f"A IA analisou a imagem e detectou classe #{predicted_class}. Isso pode ser usado para personalizar edições futuramente.")

        st.image(image, caption="Imagem Editada", use_column_width=True)

        buf = BytesIO()
        image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Baixar imagem editada",
            data=byte_im,
            file_name="imagem_editada.jpg",
            mime="image/jpeg"
        )

    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
else:
    st.info("Por favor, envie uma imagem para começar.")
