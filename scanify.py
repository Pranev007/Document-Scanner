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

    # Upload Section (Multiple files enabled)
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

# ---------------------- IMAGE PROCESSING ----------------------

def preprocess_image(img):
    img = cv2.resize(img, (960, 1280), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(enhanced, 11, 75, 75)
    edges = cv2.Canny(blurred, 30, 120)
    return img, blurred, edges

def get_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5000 < area < 1000000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 0.7 < aspect_ratio < 1.3:
                    if area > max_area:
                        max_area = area
                        best_cnt = approx
    return best_cnt.reshape(4, 2) if best_cnt is not None else None

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    return rect

def warp_perspective(img, pts):
    ordered_pts = order_points(pts)
    tl, tr, br, bl = ordered_pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_width = int(max(widthA, widthB) * 1.05)
    max_height = int(max(heightA, heightB) * 1.05)
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped

def enhance_document(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_planes = cv2.split(rgb)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm)
    result = cv2.merge(result_planes)
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 10)

def auto_crop(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([255, 60, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = (0, 0, img.shape[1], img.shape[0])
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        width = rect[1][0]
        height = rect[1][1]
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 1.5 and area > max_area:
            max_area = area
            x, y, w, h = cv2.boundingRect(box)
            best_rect = (x, y, w, h)
    x, y, w, h = best_rect
    return img[y:y + h, x:x + w]

def process_image(img):
    _, blurred, edges = preprocess_image(img)
    biggest = get_contours(edges)
    if biggest is not None:
        warped = warp_perspective(blurred, biggest)
        cropped = auto_crop(warped)
        enhanced = enhance_document(cropped)
    else:
        cropped = auto_crop(img)
        enhanced = enhance_document(cropped)
    return enhanced

if __name__ == "__main__":
    main()




