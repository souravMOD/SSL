import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import glob
import os
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
import time

def calculate_similarity(image1, image2, downscale=0.5):
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    if downscale < 1:
        image1 = cv2.resize(image1, (int(image1.shape[1] * downscale), int(image1.shape[0] * downscale)))
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(gray1, gray2, full=True)
    return score

def align_images(base_img, img_to_align, max_keypoints=500):
    base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    align_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    keypoints1, descriptors1 = orb.detectAndCompute(base_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(align_gray, None)
    
    if descriptors1 is None or descriptors2 is None:
        return img_to_align
    
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    if len(good_matches) < 4:
        return img_to_align
    
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    height, width = base_img.shape[:2]
    aligned_image = cv2.warpPerspective(img_to_align, matrix, (width, height))
    
    return aligned_image

def create_video_from_frames(frames, output_filename, fps=10):
    # Ensure all frames are the same size
    standard_size = (frames[0].shape[1], frames[0].shape[0])  # width, height from the first frame
    resized_frames = [
        cv2.resize(frame, standard_size) if frame.shape[:2] != standard_size[::-1] else frame for frame in frames
    ]

    # Convert each frame from BGR to RGB
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized_frames]

    # Create video
    clip = ImageSequenceClip(rgb_frames, fps=fps)
    clip.write_videofile(output_filename, codec="libx264")


# Folder containing the images
image_folder = r'D:\GenV2\train\images'
output_video = r'ordered_aligned_video.mp4'

start_time = time.time()
image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
images = [cv2.imread(img) for img in image_files if cv2.imread(img) is not None]

# Checkpoint for images loaded
print(f"[{time.time() - start_time:.2f}s] Loaded {len(images)} images for processing.")

if not images:
    raise ValueError("No images loaded. Check the folder path or file extensions.")

downscale_start = time.time()
downscaled_images = [cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))) for img in images]
print(f"[{time.time() - downscale_start:.2f}s] Downscaled images for faster processing.")

similarity_matrix = np.zeros((len(images), len(images)))

# Compute similarities with logging and timing
print("Starting similarity calculations...")
similarity_start = time.time()
with Parallel(n_jobs=-1) as parallel:
    results = parallel(
        delayed(calculate_similarity)(downscaled_images[i], downscaled_images[j]) 
        for i in range(len(images)) 
        for j in range(i + 1, len(images))
    )
print(f"[{time.time() - similarity_start:.2f}s] Completed similarity calculations.")

# Populate similarity matrix
index = 0
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        similarity_matrix[i, j] = results[index]
        similarity_matrix[j, i] = similarity_matrix[i, j]
        index += 1
print(f"[{time.time() - similarity_start:.2f}s] Similarity matrix populated.")

similarity_threshold = 0.7
print("Starting DBSCAN clustering...")
clustering_start = time.time()
clustering = DBSCAN(eps=1 - similarity_threshold, min_samples=1, metric="precomputed")
labels = clustering.fit_predict(1 - similarity_matrix)
print(f"[{time.time() - clustering_start:.2f}s] Completed clustering.")

sorted_indices = sorted(range(len(labels)), key=lambda x: labels[x])
ordered_images = [images[idx] for idx in sorted_indices]
print(f"[{time.time() - clustering_start:.2f}s] Ordered images according to similarity clusters.")

alignment_start = time.time()
aligned_images = [ordered_images[0]]
for i in tqdm(range(1, len(ordered_images)), desc="Aligning images"):
    aligned_img = align_images(aligned_images[-1], ordered_images[i], max_keypoints=300)
    aligned_images.append(aligned_img)
print(f"[{time.time() - alignment_start:.2f}s] Completed image alignment.")

print("Creating video...")
video_start = time.time()
create_video_from_frames(aligned_images, output_video, fps=30)
print(f"[{time.time() - video_start:.2f}s] Video created successfully at {output_video}")
print(f"Total execution time: {time.time() - start_time:.2f}s")
