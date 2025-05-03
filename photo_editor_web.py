import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

st.set_page_config(page_title="Editor de Fotos IA", layout="centered")
st.title("Editor de Fotos com Efeitos Dinâmicos")
st.markdown("Otimize, edite e aplique filtros em suas imagens com fluidez no Android e desktop.")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Imagem Original", use_column_width=True)
        st.markdown("---")
        st.subheader("Ajustes Personalizados")

        col1, col2 = st.columns(2)
        with col1:
            brightness = st.slider("Brilho", 0.1, 2.0, 1.0)
            contrast = st.slider("Contraste", 0.1, 2.0, 1.0)
        with col2:
            sharpness = st.slider("Nitidez", 0.1, 2.0, 1.0)
            filter_option = st.selectbox("Filtro", ["Nenhum", "Contorno", "Detalhe", "Bordas"])

        # Aplicar melhorias
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Sharpness(image).enhance(sharpness)

        if filter_option == "Contorno":
            image = image.filter(ImageFilter.CONTOUR)
        elif filter_option == "Detalhe":
            image = image.filter(ImageFilter.DETAIL)
        elif filter_option == "Bordas":
            image = image.filter(ImageFilter.FIND_EDGES)

        st.image(image, caption="Imagem Editada", use_column_width=True)

        from io import BytesIO
        buf = BytesIO()
        image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button("Baixar imagem editada", data=byte_im, file_name="imagem_editada.jpg", mime="image/jpeg")

    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
else:
    st.info("Por favor, carregue uma imagem no formato JPG ou PNG.")
