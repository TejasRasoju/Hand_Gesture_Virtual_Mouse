# utils.py
import numpy as np

def normalize_landmarks_xyz(coords):
    """
    coords: flat list or array of length 63 (x,y,z for 21 landmarks) OR length 42 (x,y only).
    Returns:
      norm_flat: normalized flattened vector (same length)
      index_xy: (x,y) of index fingertip in original image coordinates (if available)
    Normalization:
      - reshape to (21,3)
      - subtract wrist (landmark 0)
      - scale by max absolute value (hand size) to make size-invariant
    """
    arr = np.array(coords, dtype=np.float32)
    if arr.size == 0:
        return None, None
    if arr.size not in (42, 63):
        raise ValueError(f"Unexpected landmark length: {arr.size}")
    has_z = (arr.size == 63)
    if has_z:
        pts = arr.reshape(21, 3)
    else:
        pts2 = arr.reshape(21, 2)
        # pad z=0
        pts = np.zeros((21,3), dtype=np.float32)
        pts[:, :2] = pts2

    # original index fingertip absolute coordinates (x,y) from normalized image coords (0..1)
    index_xy = (float(pts[8,0]), float(pts[8,1]))

    # make wrist (0) the origin
    pts_rel = pts - pts[0:1, :]  # broadcasts

    # scale by max abs value for invariance
    max_val = np.max(np.abs(pts_rel))
    if max_val < 1e-6:
        max_val = 1.0
    pts_rel = pts_rel / max_val

    return pts_rel.flatten().tolist(), index_xy
