import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Custom CSS
st.markdown("""
<style>
    /* Modern Font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    
    /* Main Container */
    .main-container {
        max-width: 1200px;
        padding: 2rem 1rem;
        margin: 0 auto;
    }
    
    /* Header Section */
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
    
    /* Modified Upload Section */
    .upload-container {
        background: transparent !important;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Buttons */
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
    
    /* Progress Spinner */
    .stSpinner > div {
        border-color: #3498db transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

#-----------------STREAMLIT UI----------------
def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="header">
            <h1 class="title">📄 Scanify </h1>
            <p class="subheader">Professional Document Scanning Solution</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    uploaded_file = st.file_uploader(" ", 
                                   type=["jpg", "png", "jpeg"],
                                   help="Upload a document image",
                                   label_visibility="collapsed",
                                   key="main_uploader")
    
    if uploaded_file:
        with st.spinner('🔍 Scanning document...'):
            try:
                # Process image
                image = Image.open(uploaded_file).convert('RGB')
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                scanned = process_image(img)
                
                # Results Columns
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.image(image, use_container_width=True)
                    st.markdown('<p style="text-align: center; margin: 1rem 0; color: #2c3e50; font-weight: 500;">Original Document</p>', 
                              unsafe_allow_html=True)
                
                with col2:
                    st.image(scanned, use_container_width=True, clamp=True)
                    st.markdown('<p style="text-align: center; margin: 1rem 0; color: #2c3e50; font-weight: 500;">Scanned Result</p>', 
                              unsafe_allow_html=True)
                
                # Download Section
                st.markdown("---")
                st.markdown("### 📥 Download Your Scan")
                st.download_button(
                    label="Download High-Quality Scan",
                    data=cv2.imencode('.png', scanned)[1].tobytes(),
                    file_name="professional_scan.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"⚠️ Processing Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #7f8c8d; margin-top: 3rem;">
            <p>Scanify • v1.0 • Professional Document Scanning Solution</p>
            <p>Powered by OpenCV & Streamlit</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


#--------------------IMAGE PROCESSING-----------------------

def preprocess_image(img):
    """Enhanced preprocessing with dynamic thresholding"""
    img = cv2.resize(img, (960, 1280), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Improved adaptive thresholding
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(enhanced, 11, 75, 75)
    edges = cv2.Canny(blurred, 30, 120)  # Adjusted thresholds
    return img, blurred, edges



def get_contours(edges):
    """Improved contour detection with aspect ratio validation"""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5000 < area < 1000000:  # Dynamic area range
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4:
                x,y,w,h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 0.7 < aspect_ratio < 1.3:  # Validate document shape
                    if area > max_area:
                        max_area = area
                        best_cnt = approx
    
    return best_cnt.reshape(4,2) if best_cnt is not None else None



def order_points(pts):
    """Order contour points consistently"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[3] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[2] = pts[np.argmax(diff)]  # Bottom-left
    return rect



def warp_perspective(img, pts):
    """Improved perspective transform with edge padding"""
    ordered_pts = order_points(pts)
    tl, tr, br, bl = ordered_pts
    
    # Calculate new dimensions with 5% padding
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    
    max_width = max(int(widthA), int(widthB)) * 1.05
    max_height = max(int(heightA), int(heightB)) * 1.05
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(img, M, (int(max_width), int(max_height)))
    return warped



def enhance_document(img):
    """Fixed shadow removal with proper OpenCV constants"""
    # Convert to RGB for better color processing
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_planes = cv2.split(rgb)
    result_planes = []
    
    for plane in rgb_planes:
        # Improved shadow removal
        dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        
        # Fixed normalization parameters
        norm = cv2.normalize(diff, None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm)
    
    # Merge and convert to grayscale
    result = cv2.merge(result_planes)
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    
    # Optimized adaptive thresholding
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 21, 10)



def auto_crop(img):
    """Enhanced cropping that only keeps document content"""
    # Convert to HSV for better background detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([255, 60, 255]))
    
    # Find contours with strict document validation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = (0, 0, img.shape[1], img.shape[0])  # Default to full image
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Minimum document size
            continue
            
        # Get rotated rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Calculate aspect ratio
        width = rect[1][0]
        height = rect[1][1]
        aspect_ratio = max(width, height) / min(width, height)
        
        if aspect_ratio < 1.5 and area > max_area:  # Validate document shape
            max_area = area
            x,y,w,h = cv2.boundingRect(box)
            best_rect = (x, y, w, h)

    # Apply final crop
    x, y, w, h = best_rect
    return img[y:y+h, x:x+w]




def process_image(img):
    """Optimized pipeline with strict cropping"""
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




