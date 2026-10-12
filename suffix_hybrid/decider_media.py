"""d3 server-side media decoding: strict base64 images and videos -> vLLM inputs.

Ported from vllm-sr/d3 ``d3_runtime.py`` (revision 5ce995df, Apache-2.0):
``load_image(strict=True)``, ``_decode_video``, ``video_sample_indices`` and
``configure_video_processor``. Pixels are decoded here; resizing is left to the
checkpoint's processor inside vLLM (``D3_MM_PROCESSOR_KWARGS``), as d3 does.
"""
import base64
import binascii
import io
import math
import os
import tempfile

IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
MAX_IMAGE_BYTES = 8_000_000
MAX_IMAGE_SOURCE_PIXELS = 16_000_000
VIDEO_FORMATS = ("mp4", "webm", "quicktime", "x-matroska")
MAX_VIDEO_BYTES = 32_000_000
MAX_VIDEO_SOURCE_PIXELS = 8_294_400
MAX_VIDEO_SECONDS = 300
VIDEO_FPS = 2.0
VIDEO_MIN_FRAMES = 4
VIDEO_MAX_FRAMES = 32
VIDEO_MIN_PIXELS = 4_096
VIDEO_MAX_PIXELS = 200_704
VIDEO_MAX_TOKENS = 16_384
VIDEO_SOURCE_FPS = (0.1, 1000.0)
VIDEO_UNKNOWN_FPS = 24.0

# d3_runtime load_processor / configure_video_processor as vLLM Qwen3-VL
# mm_processor_kwargs; the pool must serve exactly these (contract-checked).
D3_MM_PROCESSOR_KWARGS = {
    "images_kwargs": {"size": {"shortest_edge": 65_536, "longest_edge": 1_638_400}},
    "videos_kwargs": {"size": {"shortest_edge": VIDEO_MIN_PIXELS,
                               "longest_edge": VIDEO_MAX_PIXELS * VIDEO_MAX_FRAMES},
                      "cap_pixels_per_frame": True,
                      "max_video_tokens": VIDEO_MAX_PIXELS // 32**2,
                      "do_sample_frames": False},
}


def _payload(value, kind, formats, max_bytes):
    header, separator, encoded = value.partition(",") if isinstance(value, str) else ("", "", "")
    header = header.strip().lower()
    if (not separator or not header.startswith(f"data:{kind}/")
            or not header.endswith(";base64")
            or header[len(f"data:{kind}/"):].split(";")[0] not in formats):
        raise ValueError(f"{kind}s must be base64 data URLs of: {', '.join(formats)}")
    if len(encoded) > 4 * -(-max_bytes // 3):
        raise ValueError(f"each {kind} must be at most {max_bytes:,} bytes")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except binascii.Error as error:
        raise ValueError(f"invalid base64 {kind} data") from error
    if len(payload) > max_bytes:
        raise ValueError(f"each {kind} must be at most {max_bytes:,} bytes")
    return payload


def load_image(value):
    """RGB PIL image; PIL decode + convert only (no EXIF rotation, no resize)."""
    from PIL import Image, UnidentifiedImageError
    payload = _payload(value, "image", ("png", "jpeg", "jpg", "webp"), MAX_IMAGE_BYTES)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in IMAGE_FORMATS:
                raise ValueError("images must be PNG, JPEG or WebP")
            if image.width * image.height > MAX_IMAGE_SOURCE_PIXELS:
                raise ValueError(f"each image must have at most {MAX_IMAGE_SOURCE_PIXELS:,} pixels")
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError(f"invalid image data ({type(error).__name__})") from error


def sample_indices(total, fps):
    """2 frames per second, at least 4 and at most 32, spread over the whole video."""
    import numpy as np
    count = min(max(int(total / fps * VIDEO_FPS), VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES, total)
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def load_video(value):
    """(uint8 [frames, h, w, 3] RGB, metadata) for vLLM, decoded by OpenCV like d3."""
    import cv2
    import numpy as np
    payload = _payload(value, "video", VIDEO_FORMATS, MAX_VIDEO_BYTES)
    try:
        api = cv2.CAP_FFMPEG if cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG) else cv2.CAP_ANY
    except (AttributeError, cv2.error):
        api = cv2.CAP_ANY
    with tempfile.NamedTemporaryFile(prefix="d3-video-", delete=False) as handle:
        handle.write(payload)
    frames, kept, reader = [], [], None
    try:
        reader = cv2.VideoCapture(handle.name, api)
        if not reader.isOpened():
            raise ValueError("the video could not be opened")
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("the video has no frame size")
        if width * height > MAX_VIDEO_SOURCE_PIXELS:
            raise ValueError(f"video frames must have at most {MAX_VIDEO_SOURCE_PIXELS:,} pixels")
        fps = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (math.isfinite(fps) and VIDEO_SOURCE_FPS[0] <= fps <= VIDEO_SOURCE_FPS[1]):
            fps = VIDEO_UNKNOWN_FPS
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:  # the container does not say: count, then read again
            limit = int(MAX_VIDEO_SECONDS * fps)
            while reader.grab():
                total += 1
                if total > limit:
                    break
            reader.release()
            reader = cv2.VideoCapture(handle.name, api)
        if total <= 0:
            raise ValueError("the video has no frames")
        if total / fps > MAX_VIDEO_SECONDS:
            raise ValueError(f"each video must be at most {MAX_VIDEO_SECONDS} seconds long")
        indices = sample_indices(total, fps)
        wanted = set(indices)
        for number in range(indices[-1] + 1):
            if not reader.grab():
                continue
            if number in wanted:
                ok, frame = reader.retrieve()
                if ok and frame is not None and frame.size:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    kept.append(number)
    except cv2.error as error:
        raise ValueError(f"invalid video data ({error.__class__.__name__})") from error
    finally:
        if reader is not None:
            reader.release()
        os.unlink(handle.name)
    if not frames:
        raise ValueError("no frame of the video could be decoded")
    if len(frames) % 2:
        frames.append(frames[-1])
        kept.append(kept[-1])
    try:
        stacked = np.ascontiguousarray(np.stack(frames))
    except ValueError as error:
        raise ValueError("all frames of a video must have the same size") from error
    metadata = {"total_num_frames": int(total), "fps": fps,
                "width": int(stacked.shape[2]), "height": int(stacked.shape[1]),
                "duration": total / fps, "frames_indices": kept, "do_sample_frames": False}
    return stacked, metadata


def video_processor(model_path):
    """The checkpoint's video processor with d3's budgets, for token planning."""
    from transformers import AutoVideoProcessor
    processor = AutoVideoProcessor.from_pretrained(model_path)
    kwargs = D3_MM_PROCESSOR_KWARGS["videos_kwargs"]
    processor.size = dict(kwargs["size"])
    processor.cap_pixels_per_frame = kwargs["cap_pixels_per_frame"]
    processor.max_video_tokens = kwargs["max_video_tokens"]
    processor.do_sample_frames = False
    return processor


def check_video_budget(processor, videos):
    merge = processor.merge_size ** 2
    planned = sum(processor.get_num_of_video_patches(len(frames), frames.shape[1], frames.shape[2])
                  for frames, _ in videos) // merge
    if planned > VIDEO_MAX_TOKENS:
        raise ValueError(f"the videos take {planned:,} input tokens, over the {VIDEO_MAX_TOKENS:,} "
                         "tokens a request may spend on videos; send fewer or shorter videos")
