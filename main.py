import cv2
from processing import preprocess_image, get_contours, warp_perspective, auto_crop, enhance_document

# Load and process image
image_path = r"C:\Users\Asus\OneDrive\Desktop\Scanify\image\image.jpg"
img, _, _, edges = preprocess_image(image_path)
biggest = get_contours(edges, img)
# Warp and crop
warped_img = warp_perspective(img, biggest)
cropped_img = auto_crop(warped_img)
final_img = enhance_document(cropped_img)  

# Display results
cv2.imshow("Scanned Document", final_img)
cv2.imwrite("final_scanned.jpg", final_img)  # Save the final document
print("Final scanned document saved as 'final_scanned.jpg'")

cv2.waitKey(0)
cv2.destroyAllWindows()
