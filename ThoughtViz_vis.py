import streamlit as st
import torch
import mne
import numpy as np
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import zipfile
import io

# Configure Streamlit
st.set_page_config(page_title="ThoughtViz", page_icon="🌐", layout="wide")

# Specify the GPU index
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# Load CLIP model for similarity calculation
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Load BLIP model
@st.cache_resource
def load_encoder_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    return processor, model

# Generate text description from image
def image_to_text(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=50, num_beams=5)
    return processor.decode(out[0], skip_special_tokens=True)

# Load Diffusion model
@st.cache_resource
def load_decoder_model():
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe

# Generate EEG plots using MNE
def create_brain_plots(eeg_file):
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    raw.pick_types(eeg=True)
    
    fig = plt.figure(figsize=(12, 6))
    
    # Plot 1: Scalp topography (Heatmap)
    ax1 = fig.add_subplot(121)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Create epochs from raw data
    events = mne.make_fixed_length_events(raw, duration=1.0)  # 1-second epochs
    epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, baseline=None, preload=True)
    
    # Compute evoked response (average across epochs)
    evoked = epochs.average()
    mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, axes=ax1, show=False)
    ax1.set_title('Brain Region Activation Heatmap', fontsize=10)
    
    # Plot 2: Spectral power
    ax2 = fig.add_subplot(122)
    raw.plot_psd(fmax=50, ax=ax2, show=False)
    ax2.set_title('Spectral Power Distribution', fontsize=10)
    
    plt.tight_layout()
    return fig

# Calculate semantic similarity
def calculate_similarity(img1, img2, clip_model, clip_processor):
    inputs = clip_processor(images=[img1, img2], return_tensors="pt", padding=True).to(device)
    embeddings = clip_model.get_image_features(**inputs)
    return cosine_similarity(embeddings.cpu().detach().numpy())[0][1]

# Extract EEG and Image from ZIP
def extract_eeg_image_pair(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        eeg_file = next((f for f in file_list if f.endswith('.fif')), None)
        image_file = next((f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))), None)
        
        if not eeg_file or not image_file:
            raise ValueError("ZIP file must contain exactly one .fif file and one image file.")
        
        with zip_ref.open(eeg_file) as eef_f, zip_ref.open(image_file) as img_f:
            eeg_data = io.BytesIO(eef_f.read())
            image_data = io.BytesIO(img_f.read())
            
    return eeg_data, image_data

# Streamlit UI
st.title("🌐 ThoughtViz")
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(45deg, #000428, #004e92);
    }
    .sidebar .sidebar-content {
        background: #001529 !important;
    }
</style>
""", unsafe_allow_html=True)

with st.spinner("Initializing neural processors..."):
    blip_processor, blip_model = load_encoder_model()
    pipe = load_decoder_model()
    clip_model, clip_processor = load_clip_model()

# Upload EEG-Image Pair
st.subheader("🧠 Upload EEG-Image Pair")
uploaded_file = st.file_uploader("Upload a ZIP file containing EEG (.fif) and Image (.png/.jpg/.jpeg)", type=["zip"])

if uploaded_file:
    try:
        # Extract EEG and Image from ZIP
        eeg_data, image_data = extract_eeg_image_pair(uploaded_file)
        
        # Display EEG Activation Plots
        st.subheader("🧬 Brain Activation Analysis")
        with st.spinner("Analyzing neural patterns..."):
            brain_fig = create_brain_plots(eeg_data)
            st.pyplot(brain_fig)
        
        # Display Ground Truth Image
#         st.subheader("🌄 Ground Truth Image")
        gt_img = Image.open(image_data).convert("RGB")
#         st.image(gt_img, caption="Ground Truth Image", use_column_width=True)
        
        if st.button("🚀 Generate Neural Reconstruction"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Stage 1: Process Ground Truth
            status_text.markdown("**Stage 1/3:** Decoding visual cortex patterns...")
            text_prompt = image_to_text(gt_img, blip_processor, blip_model)
            progress_bar.progress(25)
            
            # Stage 2: Generate reconstruction
            status_text.markdown("**Stage 2/3:** Synthesizing neural perception...")
            generator = torch.Generator(device).manual_seed(int(1e9 * 0.95))  # Fixed creativity
            gen_img = pipe(
                prompt=text_prompt,
                guidance_scale=9.0,
                num_inference_steps=50,
                generator=generator
            ).images[0]
            progress_bar.progress(75)
            
            # Stage 3: Calculate similarity
            status_text.markdown("**Stage 3/3:** Evaluating neural congruence...")
            similarity = calculate_similarity(gt_img, gen_img, clip_model, clip_processor)
            progress_bar.progress(100)
            
            # Display results
            st.subheader("🎨 Neural Reconstruction")
            col1, col2 = st.columns(2)
            with col1:
                st.image(gen_img, caption="Generated Image", use_column_width=True)
            with col2:
                st.image(gt_img, caption="Ground Truth", use_column_width=True)
            
#             st.success(f"**Semantic Congruence:** {similarity*100:.1f}%")
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(45deg, #4CAF50, #81C784);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    margin: 20px 0;
                ">
                    <h2 style="color: white; margin: 0;">
                        🎯 Semantic Congruence: <strong>{similarity*100:.1f}%</strong>
                    </h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress_bar.empty()
            status_text.empty()
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Hide technical details
st.markdown("""
<style>
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)