"""
RunPod Serverless handler for image-to-video generation using LTX-Video.
Receives base64 image + prompt, returns base64 video or upload URL.
Uses /runpod-volume/models as persistent cache to avoid repeated downloads.
"""
import os
import io
import base64
import time
import tempfile
import subprocess

import runpod
import torch
from PIL import Image

print(f"Python startup OK. torch={torch.__version__}, cuda={torch.cuda.is_available()}")

try:
    from diffusers import LTXImageToVideoPipeline
    print("diffusers LTXImageToVideoPipeline import OK")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    LTXImageToVideoPipeline = None

try:
    from diffusers.utils import export_to_video
    print("diffusers export_to_video import OK")
except ImportError:
    export_to_video = None
    print("export_to_video not available, will use imageio directly")

MODEL_ID = os.environ.get("MODEL_ID", "Lightricks/LTX-Video-0.9.1")
VOLUME_CACHE = "/runpod-volume/models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return

    cache_dir = VOLUME_CACHE if os.path.isdir("/runpod-volume") else None
    source = "volume cache" if cache_dir and os.listdir(cache_dir or "/nonexistent") else "HuggingFace"
    print(f"Loading {MODEL_ID} from {source} on {DEVICE}...")
    start = time.time()

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    pipe = LTXImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        cache_dir=cache_dir,
    ).to(DEVICE)

    pipe.enable_model_cpu_offload()

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s (source: {source})")


def handler(job):
    load_model()

    inp = job["input"]
    image_b64 = inp.get("image")
    prompt = inp.get("prompt", "")
    duration = int(inp.get("duration", 5))
    width = int(inp.get("width", 704))
    height = int(inp.get("height", 480))
    fps = int(inp.get("fps", 24))
    steps = int(inp.get("steps", 30))
    seed = inp.get("seed")

    num_frames = min(duration * fps, 257)

    if not image_b64:
        return {"error": "No image provided"}

    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.seed()

    print(f"Generating {num_frames} frames at {width}x{height}, {steps} steps...")
    gen_start = time.time()

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt="worst quality, blurry, distorted, deformed, extra limbs",
        num_frames=num_frames,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    )

    gen_elapsed = time.time() - gen_start
    print(f"Generation done in {gen_elapsed:.1f}s")

    frames = output.frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    if export_to_video:
        export_to_video(frames, tmp_path, fps=fps)
    else:
        import imageio
        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            import numpy as np
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    os.unlink(tmp_path)

    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "video_base64": video_b64,
        "duration_seconds": len(frames) / fps,
        "frames": len(frames),
        "width": width,
        "height": height,
        "fps": fps,
        "generation_time": round(gen_elapsed, 1),
    }


runpod.serverless.start({"handler": handler})
