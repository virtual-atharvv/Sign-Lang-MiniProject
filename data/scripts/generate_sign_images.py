import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define standard MediaPipe Holistic connectivity graphs for clean visualization
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # upper body arms & shoulders
    (11, 23), (12, 24), (23, 24)                      # torso
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky
]

def render_skeleton_image(keypoints_seq: np.ndarray, gloss_title: str, save_path: str):
    """
    Renders a keyframe composite of the 3D landmark sequence onto a dark canvas and saves as PNG.
    """
    T, N, C = keypoints_seq.shape
    
    # Pick 4 representative temporal keyframes across the sign duration
    keyframe_indices = np.linspace(0, T - 1, min(4, T), dtype=int)
    
    fig, axes = plt.subplots(1, len(keyframe_indices), figsize=(4 * len(keyframe_indices), 4.5), facecolor='#0f172a')
    if len(keyframe_indices) == 1:
        axes = [axes]
        
    fig.suptitle(f"ISL Sign Dataset Skeleton: {gloss_title.upper()}", color='#f8fafc', fontsize=16, fontweight='bold', y=0.98)
    
    for idx, ax in enumerate(axes):
        t = keyframe_indices[idx]
        frame = keypoints_seq[t] # (75, 3)
        
        ax.set_facecolor('#1e293b')
        ax.axis('off')
        ax.set_aspect('equal')
        
        # 1. Pose landmarks (0..32)
        pose_x = frame[:33, 0]
        pose_y = frame[:33, 1]
        
        # Flip Y for correct top-down image orientation
        ax.scatter(pose_x[11:25], -pose_y[11:25], c='#38bdf8', s=30, zorder=3, label='Pose')
        for p1, p2 in POSE_CONNECTIONS:
            if p1 < len(pose_x) and p2 < len(pose_x):
                if pose_x[p1] > 0 and pose_x[p2] > 0:
                    ax.plot([pose_x[p1], pose_x[p2]], [-pose_y[p1], -pose_y[p2]], color='#0284c7', linewidth=2.5, zorder=2)
                    
        # 2. Left Hand landmarks (33..53)
        lh_x = frame[33:54, 0]
        lh_y = frame[33:54, 1]
        if np.any(lh_x > 0.01):
            ax.scatter(lh_x, -lh_y, c='#34d399', s=25, zorder=4, label='Left Hand')
            for p1, p2 in HAND_CONNECTIONS:
                ax.plot([lh_x[p1], lh_x[p2]], [-lh_y[p1], -lh_y[p2]], color='#10b981', linewidth=1.8, zorder=3)
                
        # 3. Right Hand landmarks (54..74)
        rh_x = frame[54:75, 0]
        rh_y = frame[54:75, 1]
        if np.any(rh_x > 0.01):
            ax.scatter(rh_x, -rh_y, c='#f472b6', s=25, zorder=4, label='Right Hand')
            for p1, p2 in HAND_CONNECTIONS:
                ax.plot([rh_x[p1], rh_x[p2]], [-rh_y[p1], -rh_y[p2]], color='#ec4899', linewidth=1.8, zorder=3)
                
        ax.set_title(f"Frame {t+1}/{T}", color='#94a3b8', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    output_dir = "/home/elliot/MiniProject/data/sign_dataset_images"
    os.makedirs(output_dir, exist_ok=True)
    
    pose_root = "/home/elliot/MiniProject/data/include_poses/INCLUDE/Pose_Signs"
    pkl_files = glob.glob(f"{pose_root}/**/*.pkl", recursive=True)
    print(f"Found {len(pkl_files)} dataset pickle files.")
    
    # Process up to 50 distinct signs for visual collection
    processed_glosses = set()
    count = 0
    
    for pkl_path in pkl_files:
        # Extract folder name e.g. "3. happy" -> "happy"
        folder_name = os.path.basename(os.path.dirname(pkl_path))
        gloss_name = folder_name.split(". ")[-1].replace(" ", "_").upper()
        
        if gloss_name in processed_glosses:
            continue
            
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            kps = data["keypoints"] if isinstance(data, dict) else np.array(data)
            
            if kps.ndim == 3 and kps.shape[1] >= 75:
                img_filename = f"sign_{gloss_name}.png"
                save_path = os.path.join(output_dir, img_filename)
                render_skeleton_image(kps, gloss_name, save_path)
                print(f"[{count+1}] Rendered {gloss_name} -> {save_path}")
                processed_glosses.add(gloss_name)
                count += 1
                
                if count >= 40: # Generate 40 representative sign images
                    break
        except Exception as e:
            print(f"Skipping {pkl_path}: {e}")

    print(f"\n🎉 Successfully rendered {count} sign dataset images into '{output_dir}'.")

if __name__ == "__main__":
    main()
