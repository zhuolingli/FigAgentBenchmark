#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ALPHAXIV_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MIN_MARKDOWN_BYTES = 1000


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def fetch_markdown(arxiv_id, timeout):
    url = f"https://alphaxiv.org/abs/{arxiv_id}.md"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": ALPHAXIV_UA, "Accept": "text/markdown,text/plain;q=0.9,*/*;q=0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read()
    if len(body) < MIN_MARKDOWN_BYTES:
        raise RuntimeError(f"response too small: {len(body)} bytes")
    text = body.decode("utf-8", errors="replace").strip()
    prefix = text[:500].lower()
    if "<html" in prefix or "<!doctype html" in prefix:
        raise RuntimeError("received HTML instead of Markdown")
    return url, content_type, text


def save_success(dataset_root, entry, url, content_type, text):
    paper_dir = dataset_root / entry["folder"]
    temporary = paper_dir / "paper.md.tmp"
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(paper_dir / "paper.md")
    status = {
        "arxiv_id": entry["arxiv_id"],
        "status": "fetched",
        "source_url": url,
        "retrieval_source": "AlphaXiv full Markdown endpoint",
        "content_type": content_type,
        "characters": len(text),
    }
    write_json(paper_dir / "text_status.json", status)


def save_failure(dataset_root, entry, message, attempts):
    paper_dir = dataset_root / entry["folder"]
    write_json(paper_dir / "text_status.json", {
        "arxiv_id": entry["arxiv_id"],
        "status": "failed",
        "attempts": attempts,
        "error": message,
    })


def fetch_one(dataset_root, entry, args):
    last_error = "unknown error"
    for attempt in range(1, args.retries + 1):
        try:
            url, content_type, text = fetch_markdown(entry["arxiv_id"], args.timeout)
            save_success(dataset_root, entry, url, content_type, text)
            if args.delay:
                time.sleep(args.delay)
            return {"status": "fetched", "arxiv_id": entry["arxiv_id"], "characters": len(text)}
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < args.retries:
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    retry_after = error.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else 5.0
                else:
                    wait = min(60.0, args.retry_delay * (2 ** (attempt - 1)))
                time.sleep(wait)
    save_failure(dataset_root, entry, last_error, args.retries)
    return {"status": "failed", "arxiv_id": entry["arxiv_id"], "error": last_error}


def main():
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=script.parents[1] / "FigAgentDataset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    dataset_root = args.dataset.resolve()
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    pending = [entry for entry in manifest if not ((dataset_root / entry["paper_markdown_path"]).is_file() and (dataset_root / entry["paper_markdown_path"]).stat().st_size > 0)]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print("No pending papers")
        return

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_one, dataset_root, entry, args): entry
            for entry in pending
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result["status"] == "fetched":
                print(
                    f"[{completed}/{len(pending)}] fetched {result['arxiv_id']} chars={result['characters']}",
                    flush=True,
                )
            else:
                print(
                    f"[{completed}/{len(pending)}] failed {result['arxiv_id']}: {result['error']}",
                    flush=True,
                )

    fetched_total = sum((dataset_root / entry["paper_markdown_path"]).is_file() for entry in manifest)
    summary = {
        "papers": len(manifest),
        "attempted_this_run": len(results),
        "fetched_this_run": sum(result["status"] == "fetched" for result in results),
        "failed_this_run": sum(result["status"] == "failed" for result in results),
        "fetched_total": fetched_total,
        "remaining": len(manifest) - fetched_total,
    }
    write_json(dataset_root / "text_fetch_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
