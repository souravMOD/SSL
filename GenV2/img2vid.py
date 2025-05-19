import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import glob
import os
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from joblib import Parallel, delayed

def calculate_similarity(image1, image2):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(gray1, gray2, full=True)
    return score

def align_images(base_img, img_to_align):
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(base_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(align_gray, None)
    
    if descriptors1 is None or descriptors2 is None:
        return img_to_align
    
    # Use FLANN matcher
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return img_to_align
    
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    height, width = base_img.shape[:2]
    aligned_image = cv2.warpPerspective(img_to_align, matrix, (width, height))
    
    return aligned_image

def create_video_from_frames(frames, output_filename, fps=30):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_filename, codec="libx264")

# Folder containing the images
image_folder = r'D:\GenV2\train\images'
output_video = r'ordered_aligned_video.mp4'

image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
images = [cv2.imread(img) for img in image_files if cv2.imread(img) is not None]

if not images:
    raise ValueError("No images loaded. Check the folder path or file extensions.")

n = len(images)
similarity_matrix = np.zeros((n, n))

# Parallel similarity computation
def compute_similarity(i, j):
    return calculate_similarity(images[i], images[j])

with Parallel(n_jobs=-1) as parallel:
    results = parallel(delayed(compute_similarity)(i, j) for i in range(n) for j in range(i + 1, n))

# Fill the similarity matrix from the results
index = 0
for i in range(n):
    for j in range(i + 1, n):
        similarity_matrix[i, j] = results[index]
        similarity_matrix[j, i] = similarity_matrix[i, j]
        index += 1

# Find sequence based on maximum similarity
sequence = [0]
used = set(sequence)
for _ in tqdm(range(1, n), desc="Ordering images by similarity"):
    last = sequence[-1]
    try:
        next_image = max(
            [(j, similarity_matrix[last, j]) for j in range(n) if j not in used],
            key=lambda x: x[1]
        )[0]
    except ValueError:
        print("Error finding next image. Check similarity scores.")
        break
    sequence.append(next_image)
    used.add(next_image)

# Arrange and align images based on computed sequence
ordered_aligned_images = [images[sequence[0]]]
for i in tqdm(range(1, len(sequence)), desc="Aligning images"):
    aligned_img = align_images(ordered_aligned_images[-1], images[sequence[i]])
    ordered_aligned_images.append(aligned_img)

# Create video
create_video_from_frames(ordered_aligned_images, output_video, fps=30)
