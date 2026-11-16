import numpy as np
import pytest

from app.detect import IMGSZ, PAD_VALUE, detect, detect_tiled, letterbox, merge_tiles, nms, tile_windows


class FakeSession:
    """Stands in for onnxruntime: returns a fixed (1, 5, N) prediction."""

    class _Input:
        name = "images"

    def __init__(self, rows):
        pred = np.zeros((1, 5, max(len(rows), 1)), dtype=np.float32)
        for index, row in enumerate(rows):
            pred[0, :, index] = row
        self.pred = pred
        self.seen = []

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, feed):
        self.seen.append(feed["images"].shape)
        return [self.pred]


@pytest.mark.parametrize(
    "shape, ratio, left, top",
    [
        ((224, 224), 1.0, 0, 0),  # square: no resize, no padding
        ((100, 400), 0.56, 0, 84),  # wide: 400 -> 224 wide, 56 high, padded top and bottom
        ((400, 100), 0.56, 84, 0),  # tall: padded left and right
    ],
)
def test_letterbox_shapes_and_padding(shape, ratio, left, top):
    grey = np.full(shape, 200, dtype=np.uint8)
    tensor, got_ratio, got_left, got_top = letterbox(grey)
    assert tensor.shape == (1, 3, IMGSZ, IMGSZ)
    assert tensor.dtype == np.float32
    assert got_ratio == pytest.approx(ratio)
    assert (got_left, got_top) == (left, top)
    plane = tensor[0, 0]
    assert np.allclose(tensor[0, 0], tensor[0, 2])  # grey repeated to 3 channels
    pad = PAD_VALUE / 255.0
    if top:
        assert np.allclose(plane[:top], pad) and np.allclose(plane[-top:], pad)
        assert np.allclose(plane[top:-top], 200 / 255.0)
    if left:
        assert np.allclose(plane[:, :left], pad) and np.allclose(plane[:, -left:], pad)
        assert np.allclose(plane[:, left:-left], 200 / 255.0)
    if not (top or left):
        assert np.allclose(plane, 200 / 255.0)


def test_letterbox_rejects_colour():
    with pytest.raises(ValueError):
        letterbox(np.zeros((10, 10, 3), dtype=np.uint8))


def test_nms_removes_overlaps_and_keeps_separate_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 260, 260]], dtype=float)
    scores = np.array([0.6, 0.9, 0.5])
    keep = nms(boxes, scores, iou=0.5)
    assert keep.tolist() == [1, 2]


def test_boxes_map_back_inside_the_image():
    # Wide 100 x 400 image: letterbox ratio 0.56, content rows 84..140 of the input.
    grey = np.zeros((100, 400), dtype=np.uint8)
    session = FakeSession(
        [
            [112.0, 112.0, 224.0, 56.0, 0.9],  # exactly the content area
            [10.0, 60.0, 60.0, 80.0, 0.8],  # sticks out into the padding (top-left)
            [220.0, 150.0, 40.0, 60.0, 0.7],  # sticks out bottom-right
            [100.0, 100.0, 10.0, 10.0, 0.01],  # below min_conf
        ]
    )
    boxes, scores = detect(session, grey, min_conf=0.05)
    assert len(boxes) == 3
    assert np.all(boxes[:, [0, 2]] >= 0) and np.all(boxes[:, [0, 2]] <= 400)
    assert np.all(boxes[:, [1, 3]] >= 0) and np.all(boxes[:, [1, 3]] <= 100)
    assert boxes[0] == pytest.approx([0, 0, 400, 100], abs=1e-3)
    assert list(scores) == sorted(scores, reverse=True)


def test_tile_windows_cover_long_images_only():
    assert tile_windows(448, 224) == [(0, 448)]  # width / height <= 2: whole image
    windows = tile_windows(1000, 100)  # 10:1 -> square windows, side 100, 50% overlap
    assert windows[0] == (0, 100) and windows[1] == (50, 150)
    assert windows[-1][1] == 1000
    assert all(x1 - x0 == 100 for x0, x1 in windows)


def test_tiling_caps_the_number_of_windows():
    session = FakeSession([[112.0, 112.0, 50.0, 50.0, 0.9]])
    grey = np.zeros((100, 2000), dtype=np.uint8)  # needs 39 windows
    boxes, scores, n_windows, cut_at = detect_tiled(session, grey, max_windows=12)
    assert n_windows == 12 and len(session.seen) == 12
    assert cut_at == 11 * 50 + 100
    assert np.all(boxes[:, 2] <= cut_at)


def test_tiling_merges_a_box_that_spans_two_windows():
    windows = [(0, 300), (225, 525)]
    width = 525
    # One hyperbola from x=200 to x=400: cut by the right edge of window 0 and
    # seen in full by window 1 (which starts at 225, so it is cut there too).
    boxes = np.array(
        [
            [200.0, 10.0, 300.0, 80.0],  # window 0, touches its seam at 300
            [225.0, 12.0, 400.0, 82.0],  # window 1, touches its seam at 225
            [20.0, 120.0, 60.0, 150.0],  # unrelated box in window 0
        ]
    )
    scores = np.array([0.5, 0.7, 0.4])
    merged, merged_scores = merge_tiles(boxes, scores, [0, 1, 0], windows, width)
    assert len(merged) == 2
    assert merged[0] == pytest.approx([200.0, 10.0, 400.0, 82.0])
    assert merged_scores[0] == pytest.approx(0.7)


def test_tiling_keeps_boxes_at_different_depths_apart():
    windows = [(0, 300), (225, 525)]
    boxes = np.array([[200.0, 10.0, 300.0, 60.0], [225.0, 150.0, 400.0, 200.0]])
    merged, _ = merge_tiles(boxes, np.array([0.5, 0.6]), [0, 1], windows, 525)
    assert len(merged) == 2


def test_tiling_keeps_two_hyperbolas_30_px_apart_near_a_window_edge():
    # Image 100 px high, windows of 100 px every 50 px: (0, 100), (50, 150), (100, 200), ...
    width = 300
    windows = tile_windows(width, 100)
    assert windows[:3] == [(0, 100), (50, 150), (100, 200)]
    # Hyperbola A spans x 40-90, hyperbola B x 120-170: 30 px apart, same depth,
    # both close to the 100 px edge. Each window reports what it sees of them.
    found = [
        ([40.0, 10.0, 90.0, 50.0], 0.80, 0),  # window 0: A whole
        ([50.0, 10.0, 90.0, 50.0], 0.60, 1),  # window 1: A cut at its left edge (50)
        ([120.0, 12.0, 150.0, 52.0], 0.50, 1),  # window 1: B cut at its right edge (150)
        ([120.0, 12.0, 170.0, 52.0], 0.85, 2),  # window 2: B whole
        ([150.0, 12.0, 170.0, 52.0], 0.40, 3),  # window 3: B cut at its left edge (150)
    ]
    boxes = np.array([f[0] for f in found])
    scores = np.array([f[1] for f in found])
    tile_ids = [f[2] for f in found]
    merged, merged_scores = merge_tiles(boxes, scores, tile_ids, windows, width)
    assert len(merged) == 2
    merged = merged[np.argsort(merged[:, 0])]
    assert merged[0] == pytest.approx([40.0, 10.0, 90.0, 50.0])
    assert merged[1] == pytest.approx([120.0, 12.0, 170.0, 52.0])

