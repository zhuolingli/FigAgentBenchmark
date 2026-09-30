#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


USER_AGENT = "drawio-method-dataset/1.0 (research dataset construction)"
MIN_MARKDOWN_CHARS = 1000


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def download_source(arxiv_id, destination, timeout):
    url = f"https://arxiv.org/src/{arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                destination.write_bytes(response.read())
            return url
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 429 and attempt == 0:
                time.sleep(5)
                continue
            raise
    raise last_error


def safe_extract(archive_path, destination):
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            root = destination.resolve()
            members = archive.getmembers()
            for member in members:
                target = (destination / member.name).resolve()
                if not target.is_relative_to(root):
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            if any(not (member.isfile() or member.isdir()) for member in members):
                raise RuntimeError('links and special source archive members are not supported')
            archive.extractall(destination, filter='data')
            return
    except tarfile.ReadError:
        body = archive_path.read_bytes()
        if b"\\documentclass" not in body and b"\\begin{document}" not in body:
            raise
        (destination / "main.tex").write_bytes(body)


def choose_main_tex(source_root):
    candidates = []
    for path in source_root.rglob("*.tex"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = 0
        if "\\documentclass" in text:
            score += 1000
        if "\\begin{document}" in text:
            score += 1000
        if "\\title" in text:
            score += 100
        lowered = path.name.lower()
        if lowered in {"main.tex", "paper.tex", "manuscript.tex", "article.tex"}:
            score += 200
        if "supp" in lowered or "appendix" in lowered:
            score -= 100
        score += min(len(text), 1_000_000) / 1_000_000
        candidates.append((score, path))
    if not candidates:
        raise RuntimeError("no .tex files found")
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return candidates[0][1]


def expand_inputs(path, source_root, seen=None):
    seen = set() if seen is None else seen
    resolved = path.resolve()
    if resolved in seen or not resolved.is_relative_to(source_root.resolve()):
        return ""
    seen.add(resolved)
    text = path.read_text(encoding="utf-8", errors="replace")

    def replace(match):
        value = match.group(1).strip()
        candidate = path.parent / value
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".tex")
        if candidate.is_file():
            return expand_inputs(candidate, source_root, seen)
        return ""

    return re.sub(r"\\(?:input|include|subfile)\s*\{([^}]+)\}", replace, text)


def lightweight_latex_to_markdown(main_tex, source_root):
    text = expand_inputs(main_tex, source_root)
    document = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, flags=re.DOTALL)
    if document:
        text = document.group(1)
    text = re.sub(r"(?m)(?<!\\)%.*$", "", text)
    for environment in ("figure", "figure\\*", "table", "table\\*", "tikzpicture"):
        text = re.sub(
            rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}",
            "",
            text,
            flags=re.DOTALL,
        )
    headings = [
        ("section", "#"),
        ("subsection", "##"),
        ("subsubsection", "###"),
        ("paragraph", "####"),
    ]
    for command, marker in headings:
        text = re.sub(
            rf"\\{command}\*?(?:\[[^\]]*\])?\{{([^{{}}]*)\}}",
            lambda match: f"\n\n{marker} {match.group(1)}\n\n",
            text,
        )
    text = re.sub(r"\\item(?:\[[^\]]*\])?", "\n- ", text)
    text = re.sub(r"\\(?:label|bibliography|bibliographystyle)\s*\{[^{}]*\}", "", text)
    for _ in range(8):
        updated = re.sub(
            r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}",
            r"\1",
            text,
        )
        if updated == text:
            break
        text = updated
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    text = text.replace("~", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < MIN_MARKDOWN_CHARS:
        raise RuntimeError(f"lightweight Markdown too short: {len(text)} characters")
    return text


def convert_with_pandoc(main_tex, source_root, force_lightweight=False):
    if force_lightweight:
        markdown = lightweight_latex_to_markdown(main_tex, source_root)
        return markdown, "Used forced lightweight recursive LaTeX conversion"
    try:
        result = subprocess.run(
            ["pandoc", "--from=latex", "--to=gfm", "--wrap=none", main_tex.name],
            cwd=main_tex.parent,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        markdown = lightweight_latex_to_markdown(main_tex, source_root)
        return markdown, "Pandoc timed out; used lightweight recursive LaTeX conversion"
    if result.returncode != 0:
        markdown = lightweight_latex_to_markdown(main_tex, source_root)
        return markdown, f"Pandoc failed; used lightweight conversion: {result.stderr.strip()[-1000:]}"
    markdown = result.stdout.strip()
    if len(markdown) < MIN_MARKDOWN_CHARS:
        raise RuntimeError(f"converted Markdown too short: {len(markdown)} characters")
    return markdown, result.stderr.strip()


def save_success(dataset_root, entry, source_url, main_tex, markdown, warnings):
    paper_dir = dataset_root / entry["folder"]
    retrieval_source = (
        "arXiv LaTeX source converted with lightweight parser"
        if "lightweight" in warnings.lower()
        else "arXiv LaTeX source converted with Pandoc"
    )
    temporary = paper_dir / "paper.md.tmp"
    temporary.write_text(markdown + "\n", encoding="utf-8")
    temporary.replace(paper_dir / "paper.md")
    status = {
        "arxiv_id": entry["arxiv_id"],
        "status": "fetched",
        "source_url": source_url,
        "retrieval_source": retrieval_source,
        "main_tex": main_tex,
        "characters": len(markdown),
        "pandoc_warnings": warnings[-2000:] if warnings else "",
    }
    write_json(paper_dir / "text_status.json", status)


def save_failure(dataset_root, entry, error):
    paper_dir = dataset_root / entry["folder"]
    write_json(paper_dir / "text_status.json", {
        "arxiv_id": entry["arxiv_id"],
        "status": "failed",
        "retrieval_source": "arXiv LaTeX source fallback",
        "error": f"{type(error).__name__}: {error}",
    })


def main():
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=script.parents[1] / "FigAgentDataset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--lightweight", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    missing = [entry for entry in manifest if not ((dataset_root / entry["paper_markdown_path"]).is_file() and (dataset_root / entry["paper_markdown_path"]).stat().st_size > 0)]
    if args.limit is not None:
        missing = missing[:args.limit]
    fetched = 0
    failed = 0
    for position, entry in enumerate(missing, start=1):
        try:
            with tempfile.TemporaryDirectory(prefix=f"arxiv_{entry['arxiv_id']}_") as temporary:
                temporary_root = Path(temporary)
                archive_path = temporary_root / "source"
                source_root = temporary_root / "extracted"
                source_root.mkdir()
                source_url = download_source(entry["arxiv_id"], archive_path, args.timeout)
                safe_extract(archive_path, source_root)
                main_tex = choose_main_tex(source_root)
                markdown, warnings = convert_with_pandoc(main_tex, source_root, args.lightweight)
                save_success(
                    dataset_root,
                    entry,
                    source_url,
                    main_tex.relative_to(source_root).as_posix(),
                    markdown,
                    warnings,
                )
            fetched += 1
            print(f"[{position}/{len(missing)}] fetched {entry['arxiv_id']} chars={len(markdown)}", flush=True)
        except Exception as error:
            failed += 1
            save_failure(dataset_root, entry, error)
            print(f"[{position}/{len(missing)}] failed {entry['arxiv_id']}: {type(error).__name__}: {error}", flush=True)
        time.sleep(args.delay)
    print(json.dumps({"attempted": len(missing), "fetched": fetched, "failed": failed}, sort_keys=True))


if __name__ == "__main__":
    main()
