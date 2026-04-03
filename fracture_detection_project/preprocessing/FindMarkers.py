# ==============================================================
# FindMarkers.py
# FRACTURE DETECTION PROJECT
# ==============================================================
# Finds annotation letter locations (R, L, RY) in all images
# using OpenCV — no API calls, no external models
# Saves results to marker_coords.csv
# Fully resumable — safe if PC turns off
# ==============================================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# CONFIG
# ==============================================================

BASE_PATH   = r"d:\fracture_detection_project"
TRAIN_CSV   = os.path.join(BASE_PATH, "data", "train_labels.csv")
VAL_CSV     = os.path.join(BASE_PATH, "data", "valid_labels.csv")

# Output — marker locations per image
MARKER_CSV  = os.path.join(BASE_PATH, "data",
                           "marker_coords.csv")

# Checkpoint — saves progress every N images
SAVE_EVERY  = 500

# ==============================================================
# FIND LETTER MARKER IN ONE IMAGE
# ==============================================================

def find_marker(image_path: str) -> dict:
    """
    Finds annotation marker (R, L, RY, arrows)
    in an X-ray image using OpenCV.

    How it works:
    ─────────────
    Annotation markers in MURA are:
    → Small isolated bright regions
    → Rectangular/text shaped
    → Usually in corners or edges
    → NOT connected to bone structure

    Algorithm:
    1. Load grayscale
    2. Adaptive threshold → finds bright regions
    3. Connected components → finds isolated blobs
    4. Filter by size + shape → text-like blobs
    5. Return bounding box of detected marker

    Returns:
        dict with x1,y1,x2,y2 as fractions (0-1)
        OR None if no marker found
    """
    try:
        img = cv2.imread(image_path,
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        h, w = img.shape

        # ── Method 1: High brightness threshold ───────────────
        # Letters are often very bright white
        _, bright_mask = cv2.threshold(
            img, 200, 255, cv2.THRESH_BINARY)

        # ── Method 2: Adaptive threshold ──────────────────────
        # Catches letters that are bright relative
        # to their surroundings (not absolute)
        adaptive_mask = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=-20)

        # Combine both masks
        combined = cv2.bitwise_or(
            bright_mask, adaptive_mask)

        # Clean up noise
        kernel  = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(
            combined, cv2.MORPH_OPEN, kernel)

        # Find connected components
        num, labels, stats, _ = \
            cv2.connectedComponentsWithStats(
                cleaned, connectivity=8)

        best_marker = None
        best_score  = 0

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            x    = stats[i, cv2.CC_STAT_LEFT]
            y    = stats[i, cv2.CC_STAT_TOP]
            bw   = stats[i, cv2.CC_STAT_WIDTH]
            bh   = stats[i, cv2.CC_STAT_HEIGHT]

            # ── Marker constraints ─────────────────────────────

            # 1. Size: small but not noise
            min_area = 15
            max_area = 0.025 * h * w  # max 2.5% of image
            if not (min_area < area < max_area):
                continue

            # 2. Shape: text-like aspect ratio
            aspect = bw / (bh + 1e-5)
            if not (0.15 < aspect < 8.0):
                continue

            # 3. Location: markers are usually
            #    in outer 25% of image
            cx = (x + bw/2) / w
            cy = (y + bh/2) / h
            in_border = (cx < 0.25 or cx > 0.75 or
                         cy < 0.25 or cy > 0.75)

            # 4. Brightness: marker pixels should
            #    be significantly brighter than
            #    image mean
            region      = img[y:y+bh, x:x+bw]
            region_mean = float(region.mean())
            img_mean    = float(img.mean())
            bright_ratio = region_mean / (img_mean + 1e-5)

            # Score: prefer bright + border markers
            score = bright_ratio
            if in_border:
                score *= 2.0

            if score > best_score and score > 1.3:
                best_score  = score
                best_marker = {
                    "x1": round(max(0, x-5) / w, 4),
                    "y1": round(max(0, y-5) / h, 4),
                    "x2": round(min(w, x+bw+5) / w, 4),
                    "y2": round(min(h, y+bh+5) / h, 4),
                }

        return best_marker

    except Exception:
        return None


# ==============================================================
# PROCESS DATASET
# ==============================================================

def process_dataset(csv_path: str,
                    split_name: str,
                    existing_paths: set) -> list:
    """
    Processes all images in one CSV split.
    Skips already processed images.
    Returns list of result dicts.
    """
    df      = pd.read_csv(csv_path)
    results = []

    # Filter already done
    todo = df[~df["image_path"].isin(existing_paths)]
    done = len(df) - len(todo)

    print(f"\n{split_name}: {len(df)} total")
    print(f"  Already done : {done}")
    print(f"  To process   : {len(todo)}")

    if len(todo) == 0:
        print(f"  ✅ All done!")
        return results

    found_count = 0
    for _, row in tqdm(todo.iterrows(),
                       total=len(todo),
                       desc=f"  {split_name}"):
        marker = find_marker(row["image_path"])

        result = {
            "image_path" : row["image_path"],
            "label"      : row["label"],
            "body_part"  : row.get("body_part", ""),
            "has_marker" : 1 if marker else 0,
            "x1"         : marker["x1"] if marker else 0.0,
            "y1"         : marker["y1"] if marker else 0.0,
            "x2"         : marker["x2"] if marker else 0.0,
            "y2"         : marker["y2"] if marker else 0.0,
        }
        results.append(result)
        if marker:
            found_count += 1

    print(f"\n  Markers found: {found_count}/{len(todo)}"
          f" ({100*found_count/max(len(todo),1):.1f}%)")
    return results


# ==============================================================
# MAIN — with full resume support
# ==============================================================

if __name__ == "__main__":

    print("="*55)
    print("FIND MARKERS — OpenCV Letter Detection")
    print("Resumable — safe to restart anytime")
    print("="*55)

    # ── Load existing progress ────────────────────────────────
    if os.path.exists(MARKER_CSV):
        existing_df    = pd.read_csv(MARKER_CSV)
        existing_paths = set(existing_df["image_path"])
        print(f"\nResuming — already processed: "
              f"{len(existing_paths)} images")
    else:
        existing_df    = pd.DataFrame()
        existing_paths = set()
        print("\nStarting fresh")

    all_results = []

    # ── Process train set ─────────────────────────────────────
    train_results = process_dataset(
        TRAIN_CSV, "TRAIN", existing_paths)
    all_results.extend(train_results)

    # Save after train
    if all_results:
        new_df = pd.DataFrame(all_results)
        if not existing_df.empty:
            combined = pd.concat(
                [existing_df, new_df],
                ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(MARKER_CSV, index=False)
        existing_paths = set(combined["image_path"])
        existing_df    = combined
        all_results    = []
        print(f"\n✅ Train saved → {MARKER_CSV}")

    # ── Process val set ───────────────────────────────────────
    val_results = process_dataset(
        VAL_CSV, "VALIDATION", existing_paths)
    all_results.extend(val_results)

    # Final save
    if all_results:
        new_df = pd.DataFrame(all_results)
        if not existing_df.empty:
            combined = pd.concat(
                [existing_df, new_df],
                ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(MARKER_CSV, index=False)
    else:
        combined = existing_df

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("COMPLETE")
    print("="*55)
    print(f"Total images  : {len(combined)}")
    print(f"Markers found : "
          f"{combined['has_marker'].sum()}"
          f" ({100*combined['has_marker'].mean():.1f}%)")
    print(f"No marker     : "
          f"{(combined['has_marker']==0).sum()}")
    print(f"\nSaved to: {MARKER_CSV}")

    print("\nPer body part:")
    for bp in combined["body_part"].unique():
        if not bp:
            continue
        bp_df  = combined[combined["body_part"]==bp]
        pct    = bp_df["has_marker"].mean() * 100
        print(f"  {bp:<15} "
              f"{bp_df['has_marker'].sum()}"
              f"/{len(bp_df)} "
              f"({pct:.1f}% have markers)")

    print("\nNext step:")
    print("  python models/Phase1_LoRA_Exclude.py")