"""Detector: letterbox, ONNX inference, NMS, boxes back to image pixels, tiling.

The first block is vendored unchanged from GPRadar (06f), so a single-window
run gives the same boxes as the research code. Tiling is added below it.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---- Vendored from hectorvh/GPRadar src/06f_export_onnx.py at commit 1b052df
# ---- (letterbox, nms, detect and their helpers; unchanged)

IMGSZ = 224
PAD_VALUE = 114
IOU = 0.5


def letterbox(grey, size=IMGSZ):
    """Resize a greyscale image into a padded square model input.

    Keeps the aspect ratio (cv2.INTER_LINEAR), pads with 114 and centres the
    image with left = round(dw - 0.1), top = round(dh - 0.1), as Ultralytics
    does. Returns (1x3xSxS float32 in [0, 1], ratio, left, top).
    """

    grey = np.asarray(grey)
    if grey.ndim != 2:
        raise ValueError(f"expected a 2D greyscale image, got shape {grey.shape}")
    h, w = grey.shape
    ratio = min(size / h, size / w)
    new_w, new_h = max(round(w * ratio), 1), max(round(h * ratio), 1)
    if (new_w, new_h) != (w, h):
        grey = cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        grey, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_VALUE
    )
    tensor = np.repeat(padded[None, None].astype(np.float32), 3, axis=1) / 255.0
    return tensor, ratio, left, top


def _box_iou_one_to_many(box, boxes):
    """IoU of one xyxy box with an (N, 4) array of xyxy boxes."""

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-12)


def nms(boxes, scores, iou=IOU):
    """Greedy NMS on xyxy boxes; return kept indices, highest score first."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    keep = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        order = rest[_box_iou_one_to_many(boxes[best], boxes[rest]) <= iou]
    return np.asarray(keep, dtype=np.int64)


def _decode(output, min_conf):
    """Turn the (1, 5, N) output into xyxy input-pixel boxes and scores."""

    pred = np.asarray(output)[0].T
    pred = pred[pred[:, 4] > min_conf]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, pred[:, 4]


def _axis_gains(shape, ratio):
    """Per-axis scale (x, y) of the rounded resized image, as Ultralytics uses."""

    h, w = shape
    return max(round(w * ratio), 1) / w, max(round(h * ratio), 1) / h


def detect(session, grey, min_conf=0.05, iou=IOU):
    """Run the ONNX model on a greyscale image.

    Returns (boxes, scores): xyxy boxes in original-image pixels, clipped to
    the image, and their scores, highest first.
    """

    grey = np.asarray(grey)
    tensor, ratio, left, top = letterbox(grey)
    name = session.get_inputs()[0].name
    output = session.run(None, {name: tensor})[0]
    boxes, scores = _decode(output, min_conf)
    keep = nms(boxes, scores, iou=iou)
    boxes, scores = boxes[keep], scores[keep]
    gain_x, gain_y = _axis_gains(grey.shape, ratio)
    boxes = (boxes - np.array([left, top, left, top], dtype=boxes.dtype)) / np.array(
        [gain_x, gain_y, gain_x, gain_y], dtype=boxes.dtype
    )
    h, w = grey.shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes, scores


# ---- end of vendored block


# ---- Tiling (GPRadar-API, plan section 2.4)
#
# If width / height <= 2 the whole image is detected at once. Otherwise square
# windows (side = height) slide along the profile with 50% overlap, each is
# detected on its own and the boxes are shifted to full-image pixels. A box cut
# by a window edge is first joined with the matching box from the neighbouring
# window (union), then NMS (IoU 0.5) removes the duplicates the overlap makes.
# At most ``max_windows`` windows are used; a longer profile is cut and the
# caller adds a warning.

TILE_MAX_ASPECT = 2.0
TILE_OVERLAP = 0.5
SEAM_PX = 2.0
SEAM_MIN_Y_OVERLAP = 0.5


def tile_windows(width, height, max_aspect=TILE_MAX_ASPECT, overlap=TILE_OVERLAP):
    """Column ranges [(x0, x1), ...] covering ``width``; one range if no tiling."""

    width, height = int(width), int(height)
    if width <= max_aspect * height:
        return [(0, width)]
    side = max(1, min(width, height))
    stride = max(1, int(round(side * (1.0 - overlap))))
    starts = list(range(0, width - side + 1, stride))
    if starts[-1] + side < width:
        starts.append(width - side)
    return [(s, s + side) for s in starts]


def _y_overlap(a, b):
    """1-D IoU of two boxes along y."""

    inter = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(a[3], b[3]) - min(a[1], b[1])
    return inter / union if union > 0 else 0.0


def _on_seam(box, window, width, seam_px=SEAM_PX):
    """True if the box is cut by an inner edge of its window."""

    x0, x1 = window
    return (x0 > 0 and box[0] <= x0 + seam_px) or (x1 < width and box[2] >= x1 - seam_px)


def merge_tiles(boxes, scores, tile_ids, windows, width, iou=IOU, seam_px=SEAM_PX, min_y_overlap=SEAM_MIN_Y_OVERLAP):
    """Join boxes split by a window edge, then NMS. Boxes are in full-image pixels.

    Two boxes from different windows are joined when at least one is cut by
    its window's edge, they overlap (or touch) along x, and their y extents
    overlap by ``min_y_overlap`` (1-D IoU). A joined box is the union, with the
    higher score. Returns (boxes, scores), highest score first.
    """

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seam = [_on_seam(boxes[i], windows[tile_ids[i]], width, seam_px) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if tile_ids[i] == tile_ids[j] or not (seam[i] or seam[j]):
                continue
            a, b = boxes[i], boxes[j]
            if min(a[2], b[2]) - max(a[0], b[0]) < -seam_px:
                continue
            if _y_overlap(a, b) >= min_y_overlap:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if not groups:
        return np.zeros((0, 4)), np.zeros(0)
    merged_boxes = np.asarray(
        [[boxes[m, 0].min(), boxes[m, 1].min(), boxes[m, 2].max(), boxes[m, 3].max()] for m in groups.values()]
    )
    merged_scores = np.asarray([scores[m].max() for m in groups.values()])
    keep = nms(merged_boxes, merged_scores, iou=iou)
    return merged_boxes[keep], merged_scores[keep]


def detect_tiled(session, grey, min_conf=0.05, iou=IOU, max_windows=12):
    """``detect`` on the whole image, or on windows when it is long.

    Returns (boxes, scores, n_windows, cut_at): xyxy boxes in image pixels,
    highest first; ``cut_at`` is the x where analysis stopped if the window cap
    was hit, else None.
    """

    grey = np.asarray(grey)
    height, width = grey.shape
    windows = tile_windows(width, height)
    if len(windows) == 1:
        boxes, scores = detect(session, grey, min_conf=min_conf, iou=iou)
        return boxes, scores, 1, None
    cut_at = None
    if len(windows) > max_windows:
        windows = windows[:max_windows]
        cut_at = windows[-1][1]
    all_boxes, all_scores, tile_ids = [], [], []
    for index, (x0, x1) in enumerate(windows):
        boxes, scores = detect(session, np.ascontiguousarray(grey[:, x0:x1]), min_conf=min_conf, iou=iou)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] += x0
            all_boxes.append(boxes)
            all_scores.append(scores)
            tile_ids += [index] * len(boxes)
    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), len(windows), cut_at
    # Windows past a cut are absent, so the last kept edge is treated as inner too.
    boxes, scores = merge_tiles(
        np.concatenate(all_boxes), np.concatenate(all_scores), tile_ids, windows, width, iou=iou
    )
    return boxes, scores, len(windows), cut_at
