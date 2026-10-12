"""d3 media decoding: strict data URLs, d3 frame sampling, vLLM video metadata."""
import base64
import io
import sys
import tempfile
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suffix_hybrid import decider_media as media  # noqa: E402


def image_url(width, height, fmt="PNG", mime="png"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 10, 10)).save(buffer, fmt)
    return f"data:image/{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


def video_url(frames, fps, width=64, height=48):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        writer = cv2.VideoWriter(handle.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for i in range(frames):
            writer.write(np.full((height, width, 3), i % 255, np.uint8))
        writer.release()
        return "data:video/mp4;base64," + base64.b64encode(Path(handle.name).read_bytes()).decode()


def test_images_are_strict_rgb_data_urls():
    image = media.load_image(image_url(30, 20, "WEBP", "webp"))
    assert image.mode == "RGB" and image.size == (30, 20)
    for bad in ("https://example.com/a.png", image_url(4, 4, "GIF", "gif"),
                "data:image/png;base64,not base64!", image_url(4, 4, "GIF", "png")):
        with pytest.raises(ValueError):
            media.load_image(bad)


def test_sampling_matches_d3_two_fps_between_4_and_32_frames():
    assert media.sample_indices(30, 30.0) == [0, 10, 19, 29]           # 1 s -> min 4
    assert media.sample_indices(250, 25.0) == np.linspace(0, 249, 20).round().astype(int).tolist()
    assert len(media.sample_indices(24 * 40, 24.0)) == 32               # 40 s -> max 32
    assert media.sample_indices(3, 30.0) == [0, 1, 2]                   # never more than the video


def test_video_frames_are_even_with_source_indices_and_fps():
    frames, metadata = media.load_video(video_url(frames=25, fps=10))
    assert frames.dtype == np.uint8 and frames.shape[1:] == (48, 64, 3)
    assert len(frames) % 2 == 0 and len(frames) == len(metadata["frames_indices"])
    assert metadata["frames_indices"][:5] == [0, 6, 12, 18, 24] and metadata["frames_indices"][-1] == 24
    assert metadata["fps"] == 10.0 and metadata["total_num_frames"] == 25
    assert metadata["do_sample_frames"] is False
    with pytest.raises(ValueError):
        media.load_video(video_url(4, 10).replace("video/mp4", "video/avi"))


def test_processor_budgets_are_d3s():
    videos = media.D3_MM_PROCESSOR_KWARGS["videos_kwargs"]
    assert media.D3_MM_PROCESSOR_KWARGS["images_kwargs"]["size"] == {
        "shortest_edge": 65_536, "longest_edge": 1_638_400}
    assert videos["size"] == {"shortest_edge": 4_096, "longest_edge": 200_704 * 32}
    assert videos["max_video_tokens"] == 196 and videos["cap_pixels_per_frame"] is True
