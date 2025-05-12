# {
#   "apple":   ["apple1.jpg", "apple2.jpg"],
#   "banana":  ["banana.jpg"],
#   "orange":  ["orange1.jpg","orange2.jpg","orange3.jpg"],
#   # …etc
# }
# How it works
# 	1.	Feature extraction
# 	•	Use SIFT to detect keypoints & descriptors in every image.
# 	2.	Feature matching
# 	•	For each reference image, match its descriptors against the query descriptors with a Brute‐Force matcher and apply the Lowe ratio test to keep only “good” matches.
# 	3.	Score by object
# 	•	If an object has multiple reference images, sum up its good‐match counts.
# 	•	Pick the object with the highest total.
# 	4.	Thresholding (optional)
# 	•	You can reject spurious matches by requiring a minimum good‐match count.

# <><><> AnyMatch <><><>

import cv2
import numpy as np

def load_and_compute_sift(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        raise RuntimeError("SIFT is not available in your OpenCV installation. Ensure you have OpenCV with contrib modules installed.")
    kps, desc = sift.detectAndCompute(img, None)
    return kps, desc

def count_good_matches(desc1, desc2, ratio=0.75):
    # BFMatcher with L2 norm (default for SIFT)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    # Lowe’s ratio test
    good = [m for m,n in matches if m.distance < ratio * n.distance]
    return len(good)

def identify_object(query_path, refs_dict, ratio=0.75):
    # 1) Compute SIFT descriptors for query
    _, q_desc = load_and_compute_sift(query_path)
    if q_desc is None:
        raise RuntimeError("No features found in query image.")

    # 2) For each object, sum good matches across its refs
    scores = {}
    for obj, img_paths in refs_dict.items():
        total = 0
        for p in img_paths:
            _, r_desc = load_and_compute_sift(p)
            if r_desc is None: 
                continue
            total += count_good_matches(q_desc, r_desc, ratio)
        scores[obj] = total

    # 3) Find best match
    best_obj = max(scores, key=lambda k: scores[k])
    return best_obj, scores

if __name__ == "__main__":
    refs = {
      "apple":   ["apple1.jpg", "apple2.jpg"],
      "banana":  ["banana1.jpg"],
      "orange":  ["orange1.jpg","orange2.jpg"],
    }
    query = "test.jpg"
    best, all_scores = identify_object(query, refs)
    print(f"Best match: {best}")
    print("All scores:", all_scores)