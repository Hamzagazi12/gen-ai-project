import io

import numpy as np
import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# ---------------------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------------------
st.set_page_config(page_title="AI Image Variation Studio", layout="wide")
st.title("🎨 AI Image Variation & Reconstruction Studio")
st.write("Upload an image, tweak the transformation strength, and watch the AI reconstruct it!")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    st.warning(
        "No GPU detected. The app will work, but each generation can take several minutes. "
        "Use fewer steps (10-20) to speed things up."
    )


# ---------------------------------------------------------------
# 2. Load the pretrained model (cached so it loads only once)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Stable Diffusion model (first run downloads ~4 GB)...")
def load_model():
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()  # lowers memory usage
    return pipe


pipe = load_model()


# ---------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------
def calculate_mse(img1: Image.Image, img2: Image.Image) -> float:
    """Mean Squared Error between two PIL images of the same size."""
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------
# 4. Sidebar controls
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    strength = st.slider(
        "Transformation Strength",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Low = stays close to the original. High = AI changes it a lot.",
    )
    guidance = st.slider(
        "Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5,
        help="How strongly the AI follows your prompt.",
    )
    steps = st.slider(
        "Inference Steps", min_value=10, max_value=50, value=25, step=5,
        help="More steps = better quality but slower.",
    )
    seed = st.number_input("Seed (same seed = same result)", min_value=0, value=42, step=1)

# ---------------------------------------------------------------
# 5. Upload + prompts
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your input image (JPG, PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Stable Diffusion works best with RGB images sized in multiples of 8 (512x512)
    init_image = Image.open(uploaded_file).convert("RGB").resize((512, 512))
    st.image(init_image, caption="Uploaded Original Image", width=300)

    st.markdown("### Transformation Settings")
    prompt = st.text_input(
        "Transformation prompt",
        "A digital art painting of the image, fantasy style",
    )
    negative_prompt = st.text_input(
        "Negative prompt (things to avoid)",
        "blurry, low quality, distorted, deformed",
    )

    # -----------------------------------------------------------
    # 6. Generate
    # -----------------------------------------------------------
    if st.button("Generate Transformation", type="primary"):
        with st.spinner("Reconstructing image... This may take a moment."):
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator,
            )
            transformed = result.images[0]

        # Save in session state so the result survives widget changes
        st.session_state["original"] = init_image
        st.session_state["transformed"] = transformed

# ---------------------------------------------------------------
# 7. Show results
# ---------------------------------------------------------------
if "transformed" in st.session_state and uploaded_file is not None:
    original = st.session_state["original"]
    transformed = st.session_state["transformed"]

    st.markdown("### Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", use_container_width=True)
    with col2:
        st.image(transformed, caption="Transformed Image", use_container_width=True)

    st.download_button(
        "⬇️ Download Transformed Image",
        data=image_to_bytes(transformed),
        file_name="transformed.png",
        mime="image/png",
    )

    # -----------------------------------------------------------
    # 8. Metrics
    # -----------------------------------------------------------
    mse_value = calculate_mse(original, transformed)
    st.markdown("### Image Difference Metrics")
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse_value:,.2f}")

    if mse_value < 1000:
        st.info("Low MSE: The reconstruction is very close to the original.")
    else:
        st.warning("High MSE: The model has heavily modified the original image.")
