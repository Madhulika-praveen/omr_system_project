'''import cv2
import numpy as np

def deskew_image(img, max_tilt=15, hough_thresh=200, auto_orient=True):
    """Deskew the image using Hough lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        print("[INFO] No lines detected, returning original image.")
        return img, 0

    angles = []
    for rho, theta in lines[:,0]:
        angle = (theta * 180/np.pi) - 90
        # Normalize to [-90, 90]
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        angles.append(angle)

    if not angles:
        print("[INFO] No suitable angles found, returning original image.")
        return img, 0

    median_angle = float(np.median(angles))

    # Snap sideways rotation (near 90°) if auto_orient
    if auto_orient and abs(median_angle) > 45:
        median_angle = 90 if median_angle > 0 else -90
        print(f"[INFO] Detected sideways rotation. Rotating by {median_angle}°.")
    elif abs(median_angle) > max_tilt:
        print(f"[INFO] Angle {median_angle:.2f}° too large, skipping correction.")
        return img, 0
    else:
        print(f"[INFO] Correcting tilt angle: {median_angle:.2f}°")

    # Rotate image
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    rotated, angle = deskew_image(img)
    print(f"Applied rotation angle: {angle}°")

    # Save deskewed image
    cv2.imwrite("deskewed_output.jpeg", rotated)

    # Display original and deskewed images using OpenCV
    cv2.imshow("Original", img)
    cv2.imshow("Deskewed", rotated)
    cv2.waitKey(0)  # Wait until any key is pressed
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("data/images/setB/Img11.jpeg")
'''

# -----------------------------------------------------------------------------------------------
import cv2
import numpy as np

# ---------------- Deskew Image ----------------
def deskew_image(img, max_tilt=15, hough_thresh=200, auto_orient=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        print("[INFO] No lines detected, returning original image.")
        return img, 0

    angles = []
    for rho, theta in lines[:,0]:
        angle = (theta * 180/np.pi) - 90
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180
        angles.append(angle)

    if not angles:
        print("[INFO] No suitable angles found, returning original image.")
        return img, 0

    median_angle = float(np.median(angles))
    if auto_orient and abs(median_angle) > 45:
        median_angle = 90 if median_angle > 0 else -90
        print(f"[INFO] Detected sideways rotation. Rotating by {median_angle}°.")
    elif abs(median_angle) > max_tilt:
        print(f"[INFO] Angle {median_angle:.2f}° too large, skipping correction.")
        return img, 0
    else:
        print(f"[INFO] Correcting tilt angle: {median_angle:.2f}°")

    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle

# ---------------- Bubble Detection ----------------
def is_marked(bubble_roi, threshold=0.5):
    _, inv = cv2.threshold(bubble_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    fill_ratio = cv2.countNonZero(inv) / inv.size
    return fill_ratio > threshold

def extract_bubbles(sheet_image, grid_coordinates):
    detected = {}
    options = ['a', 'b', 'c', 'd']

    for subject, bubbles in grid_coordinates.items():
        detected[subject] = {}
        for q_index, start_idx in enumerate(range(0, len(bubbles), 4)):
            q_no = q_index + 1
            answer = ' '
            for opt_idx in range(4):
                try:
                    x, y, w, h = bubbles[start_idx + opt_idx]
                    roi = sheet_image[y:y+h, x:x+w]
                    if is_marked(roi):
                        answer = options[opt_idx]
                        break
                except IndexError:
                    pass
            detected[subject][q_no] = answer
    return detected

# def detect_bubble_blocks(sheet_img):
#     gray = sheet_img if len(sheet_img.shape) == 2 else cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     bubbles = []
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         if 15 < w < 35 and 15 < h < 35 and 0.8 < w/h < 1.2:
#             bubbles.append((x, y, w, h))
#     bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))
#     return bubbles
import cv2
import numpy as np

def detect_bubble_blocks(sheet_img):
    h, w = sheet_img.shape[:2]

    # ---------------- Step 1: Grayscale ----------------
    gray = sheet_img if len(sheet_img.shape) == 2 else cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)


    # ---------------- Step 2: Blur ----------------
    blur = cv2.GaussianBlur(gray, (5,5), 0)


    # ---------------- Step 3: Adaptive Threshold ----------------
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)



    # ---------------- Step 4: Morphological Cleaning ----------------
    # kernel = np.ones((2,2), np.uint8)
    # clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    clean = cv2.dilate(clean, kernel, iterations=1)  # only slight dilation


    # ---------------- Step 5: Find Contours ----------------
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_contours = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_contours, contours, -1, (0,255,0), 1)


    # ---------------- Step 6: Margins ----------------
    left_margin = int(w * 0.02)
    right_margin = int(w * 0.97)
    top_margin = int(h * 0.2)      # further lowered top margin
    bottom_margin = int(h * 0.98)    # keep bottom margin

    # Draw margins as colored lines
    margin_vis = sheet_img.copy()
    cv2.line(margin_vis, (left_margin,0), (left_margin,h), (255,0,0), 2)    # Left - Blue
    cv2.line(margin_vis, (right_margin,0), (right_margin,h), (0,0,255), 2)   # Right - Red
    cv2.line(margin_vis, (0,top_margin), (w,top_margin), (0,255,0), 2)      # Top - Green
    cv2.line(margin_vis, (0,bottom_margin), (w,bottom_margin), (0,255,255),2) # Bottom - Yellow

    # Resize only for display
    display_vis = cv2.resize(margin_vis, (0,0), fx=0.6, fy=0.6)
    cv2.imshow("Margins (Resized 0.6)", display_vis)
    cv2.waitKey(0)



    # ---------------- Step 7: Filter Bubbles ----------------
    bubbles = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh
        area = bw * bh
        area = bw * bh
        # if 14 < bw < 52 and 14 < bh < 52 and 0.65 < aspect < 1.4 and area > 250: 
        if 12 < bw < 55 and 12 < bh < 55 and 0.55 < aspect < 1.5 and area > 200:
            if left_margin < x < right_margin and top_margin < y < bottom_margin:
                bubbles.append((x, y, bw, bh))
   


    # ---------------- Step 8: Draw Filtered Bubbles ----------------
    vis_bubbles = sheet_img.copy()
    for (x,y,bw,bh) in bubbles:
        cv2.rectangle(vis_bubbles, (x,y), (x+bw, y+bh), (0,0,255), 1)

    # Scale down for display
    display_bubbles = cv2.resize(vis_bubbles, (0,0), fx=0.6, fy=0.6)
    cv2.imshow("Filtered Bubbles (Resized 0.6)", display_bubbles)
    cv2.waitKey(0)

    # ---------------- Step 9: Return Sorted Bubbles ----------------
    return sorted(bubbles, key=lambda b: (b[1], b[0]))





def generate_grid_coordinates(bubbles, questions_per_subject=20):
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
    if len(bubbles) == 0:
        return {subj: [] for subj in subjects}

    y_sorted = sorted(bubbles, key=lambda b: b[1])
    row_groups = []
    current_row = []
    row_thresh = 10

    for bubble in y_sorted:
        if not current_row:
            current_row.append(bubble)
        else:
            if abs(bubble[1] - current_row[0][1]) <= row_thresh:
                current_row.append(bubble)
            else:
                row_groups.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [bubble]
    if current_row:
        row_groups.append(sorted(current_row, key=lambda b: b[0]))

    grid_coordinates = {}
    idx = 0
    for subj in subjects:
        subject_bubbles = []
        for _ in range(questions_per_subject):
            if idx < len(row_groups):
                subject_bubbles.extend(row_groups[idx])
                idx += 1
        grid_coordinates[subj] = subject_bubbles
    return grid_coordinates

# ---------------- Main ----------------
if __name__ == "__main__":
    # Load image directly
    img_path = "data/images/setA/Img8.jpeg"  # <-- change this to your image path
    img = cv2.imread(img_path)
    if img is None:
        print("Cannot read image.")
        exit()

    # Resize image by 0.75 scaling
    # scale = 0.6
    # img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    # Deskew
    deskewed, angle = deskew_image(img)
    print(f"Applied rotation angle: {angle}°")
    cv2.imshow("Deskewed Sheet", deskewed)
    cv2.waitKey(0)

    # Detect bubbles
    bubbles = detect_bubble_blocks(deskewed)
    print(f"Detected {len(bubbles)} bubbles.")

    # Draw rectangles
    vis = deskewed.copy()
    for x, y, w, h in bubbles:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0,0,255), 1)
    cv2.imshow("Detected Bubbles", vis)
    cv2.waitKey(0)

    # Generate grid and extract answers
    grid = generate_grid_coordinates(bubbles)
    answers = extract_bubbles(deskewed, grid)
    print("Extracted answers:")
    for subj, ans in answers.items():
        print(f"{subj}: {ans}")

    # Final display
    cv2.imshow("Final Deskew + Bubbles", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------
