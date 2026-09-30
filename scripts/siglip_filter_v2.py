#!/usr/bin/env python3
"""Score diagram candidates with a pinned local SigLIP 2 model on GPU."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoTokenizer, SiglipImageProcessor

ROOT = Path(os.environ.get(
    "DRAWIO_QUALITY_ROOT",
    "work",
))
DEFAULT_INPUT = ROOT / "QUALITY_SELECTION_V2/coarse_candidates.jsonl"
DEFAULT_OUT = ROOT / "QUALITY_SELECTION_V2/model_scores.jsonl"
MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
VETO_MARGIN = -2.3

POSITIVE_PROMPTS = [
    "a polished scientific method diagram",
    "a clear system architecture diagram",
    "a complete technical workflow or flowchart",
    "a machine learning model architecture diagram",
    "a technical pipeline with connected components",
    "an algorithm process diagram",
    "a conceptual research framework diagram",
    "a technical scheduling or optimization diagram",
]

NEGATIVE_PROMPTS = [
    "a statistical bar chart",
    "a line plot or scatter plot",
    "a data table",
    "a screenshot of a software user interface",
    "a photograph or decorative illustration",
    "a mathematical plot or heatmap",
    "a repeated grid or pattern image",
    "a scientific results figure with multiple plots",
    "an ablation study or benchmark results figure",
    "a collage of disconnected figure panels",
    "a sparse unfinished or poorly formatted diagram",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def square_letterbox(image: Image.Image, max_size: int = 1024) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    side = max(resized.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(resized, ((side - resized.width) // 2, (side - resized.height) // 2))
    return canvas


def crop_content(image: Image.Image) -> Image.Image:
    mask = image.convert("L").point(lambda value: 255 if value < 250 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    margin = max(4, round(0.02 * max(right - left, bottom - top)))
    return image.crop((
        max(0, left - margin),
        max(0, top - margin),
        min(image.width, right + margin),
        min(image.height, bottom + margin),
    ))


def load_image(path: str, view_mode: str = "processor"):
    try:
        with Image.open(path) as source:
            if source.mode in ("RGBA", "LA") or "transparency" in source.info:
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(white, rgba).convert("RGB")
            else:
                image = source.convert("RGB")
            if view_mode == "processor":
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                views = [image.copy()]
            elif view_mode == "letterbox":
                views = [square_letterbox(crop_content(image))]
            else:
                views = [square_letterbox(image), square_letterbox(crop_content(image))]
            return views, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]


def aggregate(logits: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return torch.logsumexp(logits[:, start:end], dim=1) - math.log(end - start)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--load-workers", type=int, default=8)
    parser.add_argument(
        "--image-views", choices=("processor", "letterbox", "dual"), default="dual"
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.rebuild:
        print(f"scores exist, skipping: {args.output}", flush=True)
        return 0
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    device = torch.device(f"cuda:{args.gpu}")
    print(f"loading {MODEL}@{REVISION} on {device}; candidates={len(rows)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=args.local_files_only)
    image_processor = SiglipImageProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.local_files_only, torch_dtype=torch.float16
    ).eval().to(device)
    text_inputs = tokenizer(
        prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    )
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model": MODEL,
        "revision": REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "positive_prompts": POSITIVE_PROMPTS,
        "negative_prompts": NEGATIVE_PROMPTS,
        "image_views": args.image_views,
        "veto_margin": VETO_MARGIN,
        "rule": "keep unless negative aggregate exceeds positive aggregate by at least 2.3 logits",
        "model_keep_role": (
            "compatibility flag for the older conservative-veto pipeline; "
            "a calibrated final selection may override it with --margin-threshold"
        ),
    }
    (args.output.parent / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    kept = vetoed = errors = 0
    with tmp.open("w", encoding="utf-8") as fh, cf.ThreadPoolExecutor(max_workers=args.load_workers) as loader:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            loaded = list(loader.map(lambda row: load_image(row["png"], args.image_views), batch))
            good_positions = [i for i, (views, _error) in enumerate(loaded) if views is not None]
            output_rows = [None] * len(batch)
            if good_positions:
                view_count = len(loaded[good_positions[0]][0])
                images = [view for i in good_positions for view in loaded[i][0]]
                pixels = image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                inputs = {**text_inputs, "pixel_values": pixels}
                with torch.inference_mode():
                    logits = model(**inputs).logits_per_image.float()
                    logits = logits.reshape(len(good_positions), view_count, len(prompts)).mean(dim=1)
                positive = aggregate(logits, 0, len(POSITIVE_PROMPTS))
                negative = aggregate(logits, len(POSITIVE_PROMPTS), len(prompts))
                margins = positive - negative
                for local, position in enumerate(good_positions):
                    margin = float(margins[local].cpu())
                    keep = margin > VETO_MARGIN
                    kept += int(keep)
                    vetoed += int(not keep)
                    output_rows[position] = {
                        "png": batch[position]["png"],
                        "positive_logit": round(float(positive[local].cpu()), 6),
                        "negative_logit": round(float(negative[local].cpu()), 6),
                        "model_margin": round(margin, 6),
                        "model_keep": keep,
                        "model_error": "",
                        "prompt_logits": {
                            prompt: round(float(logits[local, i].cpu()), 6)
                            for i, prompt in enumerate(prompts)
                        },
                    }
            for i, row in enumerate(batch):
                if output_rows[i] is None:
                    errors += 1
                    output_rows[i] = {
                        "png": row["png"],
                        "positive_logit": None,
                        "negative_logit": None,
                        "model_margin": None,
                        "model_keep": True,
                        "model_error": loaded[i][1],
                        "prompt_logits": {},
                    }
                fh.write(json.dumps(output_rows[i], ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            done = min(start + args.batch_size, len(rows))
            if done % 512 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} kept={kept} vetoed={vetoed} errors={errors}", flush=True)
    tmp.replace(args.output)
    print(f"DONE kept={kept} vetoed={vetoed} errors={errors} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
