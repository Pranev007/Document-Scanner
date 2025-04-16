import cv2
import numpy as np

# Preprocesses the image: resize, grayscale, enhance contrast, blur, and detect edges
def preprocess_image(img):
    img = cv2.resize(img, (960, 1280), interpolation=cv2.INTER_LANCZOS4)  # Resize to standard size
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  
    enhanced = clahe.apply(gray)  
    blurred = cv2.bilateralFilter(enhanced, 11, 75, 75)  
    edges = cv2.Canny(blurred, 30, 120)  # Detect edges using Canny
    return img, blurred, edges

# Detects contours and selects the largest quadrilateral that resembles a document
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

# Orders the four corner points in a consistent order: top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  
    rect[3] = pts[np.argmax(s)]  
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[2] = pts[np.argmax(diff)]  
    return rect

# Applies a perspective warp to "flatten" the document to a top-down view
def warp_perspective(img, pts):
    ordered_pts = order_points(pts)
    tl, tr, br, bl = ordered_pts

    # Compute width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    max_width = int(max(widthA, widthB) * 1.05)
    max_height = int(max(heightA, heightB) * 1.05)

    # Define destination points for perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst) 
    warped = cv2.warpPerspective(img, M, (max_width, max_height))  
    return warped

# Enhances the document by removing shadows and binarizing the result
def enhance_document(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    rgb_planes = cv2.split(rgb)               # Split into R, G, B channels
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))  
        bg = cv2.medianBlur(dilated, 21)                       
        diff = 255 - cv2.absdiff(plane, bg)                   
        norm = cv2.normalize(diff, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)  # Normalize contrast
        result_planes.append(norm)

    result = cv2.merge(result_planes)  # Merge processed channels
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)  
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 10)  # Binarize with adaptive thresholding

# Attempts to auto-crop a low-saturation, bright document region
def auto_crop(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV
    mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([255, 60, 255]))  # Mask light gray/white areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = (0, 0, img.shape[1], img.shape[0])  # Default: full image
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
    return img[y:y + h, x:x + w]  # Crop to best bounding box

# Full document processing pipeline
def process_image(img):
    _, blurred, edges = preprocess_image(img)
    biggest = get_contours(edges) 

    if biggest is not None:
        warped = warp_perspective(blurred, biggest) 
        cropped = auto_crop(warped) 
        enhanced = enhance_document(cropped)  
    else:
        # if no contour found: try to auto-crop and enhance the original image
        cropped = auto_crop(img)
        enhanced = enhance_document(cropped)

    return enhanced 

if __name__ == "__main__":
    main()  
