
import streamlit as st
from PIL import Image, ImageEnhance
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def describe_image(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True).lower()

def choose_effect(description):
    if "nature" in description or "landscape" in description:
        return "vibrante"
    elif "person" in description or "portrait" in description:
        return "suave"
    elif "old" in description or "vintage" in description:
        return "retrô"
    elif "night" in description:
        return "preto e branco"
    else:
        return "suave"

def apply_effect(image, effect_type):
    if effect_type == "preto e branco":
        return image.convert("L").convert("RGB")
    elif effect_type == "retrô":
        return ImageEnhance.Color(image).enhance(0.6)
    elif effect_type == "vibrante":
        return ImageEnhance.Color(image).enhance(2.0)
    elif effect_type == "suave":
        return ImageEnhance.Brightness(image).enhance(1.2)
    else:
        return image

def main():
    st.set_page_config(page_title="Editor de Fotos com IA", layout="centered")
    st.title("Editor de Fotos com Inteligência Artificial")
    st.write("Selecione uma imagem e a IA aplicará o melhor efeito automaticamente.")

    uploaded_file = st.file_uploader("Carregue sua imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagem original", use_column_width=True)

        with st.spinner("Analisando imagem..."):
            processor, model = load_model()
            description = describe_image(image, processor, model)
            effect = choose_effect(description)
            edited_image = apply_effect(image, effect)

        st.success(f"Efeito aplicado: {effect.capitalize()}")
        st.image(edited_image, caption="Imagem editada", use_column_width=True)

        st.download_button(
            label="Baixar imagem editada",
            data=edited_image_to_bytes(edited_image),
            file_name="imagem_editada.jpg",
            mime="image/jpeg"
        )

def edited_image_to_bytes(img):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

if __name__ == "__main__":
    main()
