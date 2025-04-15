import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# ---------------------- Custom CSS ----------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    .main-container {
        max-width: 1200px;
        padding: 2rem 1rem;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .title {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .subheader {
        color: #7f8c8d;
        font-size: 1.1rem;
    }
    .upload-container {
        background: transparent !important;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: none !important;
        border: none !important;
    }
    .stDownloadButton button {
        background: #3498db !important;
        color: white !important;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }
    .stDownloadButton button:hover {
        background: #2980b9 !important;
        transform: translateY(-1px);
    }
    .stSpinner > div {
        border-color: #3498db transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- MAIN APP ----------------------
def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="header">
            <h1 class="title">📄 Scanify</h1>
            <p class="subheader">Professional Document Scanning Solution</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload Section 
    uploaded_files = st.file_uploader("Upload one or more images", 
                                      type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True,
                                      help="Upload document images",
                                      key="multi_uploader")

    if uploaded_files:
        scanned_images = []

        with st.spinner('🔍 Scanning documents...'):
            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    scanned = process_image(img)
                    scanned_pil = Image.fromarray(scanned).convert("RGB")
                    scanned_images.append(scanned_pil)
                except Exception as e:
                    st.error(f"⚠️ Error processing {uploaded_file.name}: {str(e)}")

        # Display results
        st.markdown("### 📄 Scanned Results")
        for i, scanned_pil in enumerate(scanned_images):
            st.image(scanned_pil, caption=f"Page {i + 1}", use_container_width=True)

        # Download as PDF
        st.markdown("---")
        st.markdown("### 📥 Download Your Scanned PDF")
        if scanned_images:
            pdf_bytes = BytesIO()
            scanned_images[0].save(pdf_bytes, format='PDF', save_all=True, append_images=scanned_images[1:])
            pdf_bytes.seek(0)

            st.download_button(
                label="Download Scanned PDF",
                data=pdf_bytes,
                file_name="scanned_document.pdf",
                mime="application/pdf"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #7f8c8d; margin-top: 3rem;">
            <p>Scanify • v1.0 • Professional Document Scanning Solution</p>
            <p>Powered by OpenCV & Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
