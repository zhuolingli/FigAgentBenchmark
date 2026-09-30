#!/usr/bin/env python3
"""Validate and export paper folders, and pack portable whole-paper TAR shards."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

ID = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
IMAGE = re.compile(r"^(.+)_p(\d{3,})\.png$")
ALLOWED = {"paper.md", "drawio", "images"}


def papers(root: Path) -> list[Path]:
    base = root / "papers" if (root / "papers").is_dir() else root
    if not base.is_dir():
        raise ValueError(f"dataset directory not found: {base}")
    with os.scandir(base) as listing:
        found = sorted(Path(e.path) for e in listing if ID.fullmatch(e.name) and e.is_dir())
    if not found:
        raise ValueError(f"no arXiv paper folders in {base}")
    return found


def file_in_paper(paper: Path, value: str, prefix: str) -> Path:
    rel = PurePosixPath(value)
    if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != prefix or '..' in rel.parts:
        raise ValueError(f"invalid asset path: {value}")
    target = paper.joinpath(*rel.parts)
    if not target.is_file():
        raise ValueError(f"missing asset: {target}")
    return target


def assets(paper: Path) -> list[tuple[Path, str]]:
    """Accept internal metadata bundles or the final data-only folders."""
    body = paper / "paper.md"
    if not body.is_file() or not body.read_text(encoding="utf-8").strip():
        raise ValueError(f"missing or empty paper.md: {paper.name}")
    result = [(body, 'paper.md')]
    draws = sorted((paper / 'drawio').glob('*.drawio'))
    if not draws:
        raise ValueError(f"no DrawIO: {paper.name}")
    result.extend((p, f'drawio/{p.name}') for p in draws)
    metadata = paper / 'metadata.json'
    if metadata.is_file():
        figures = json.loads(metadata.read_text())['figures']
        mapped = {}
        for row in figures:
            source = file_in_paper(paper, row['image_path'], 'images')
            draw = file_in_paper(paper, row['drawio_path'], 'drawio')
            page = int(row['page_number'])
            if page < 1:
                raise ValueError('page numbers must be one-based')
            name = f'images/{draw.stem}_p{page:03d}.png'
            if name in mapped and mapped[name] != source:
                raise ValueError(f'conflicting pair: {paper.name}/{name}')
            mapped[name] = source
        result.extend((p, name) for name, p in sorted(mapped.items()))
    else:
        result.extend((p, f'images/{p.name}') for p in sorted((paper/'images').glob('*.png')))
    if not any(name.startswith('images/') for _, name in result):
        raise ValueError(f'no PNG images: {paper.name}')
    return result


def validate_paper(paper: Path, *, allow_links=False, decode_pixels=16_000_000) -> dict:
    from PIL import Image
    errors, warnings = [], []
    names = {p.name for p in paper.iterdir()}
    if names != ALLOWED:
        errors.append(f'unexpected paper layout: {sorted(names)}')
    if not allow_links and (paper.is_symlink() or any(p.is_symlink() for p in paper.rglob('*'))):
        errors.append('symbolic links are not portable')
    body = paper/'paper.md'
    if not body.is_file() or not body.read_text(encoding='utf-8').strip():
        errors.append('missing or empty Markdown')
    draw_pages = {}
    for kind, suffix in [('drawio', '.drawio'), ('images', '.png')]:
        directory = paper/kind
        if not directory.is_dir():
            errors.append(f'missing {kind} directory'); continue
        entries = list(directory.iterdir())
        if not entries:
            errors.append(f'empty {kind} directory')
        for f in entries:
            if not f.is_file() or f.suffix != suffix:
                errors.append(f'unexpected asset: {kind}/{f.name}'); continue
            if kind == 'drawio':
                try:
                    with f.open('rb', buffering=1024*1024) as handle:
                        tree = ET.parse(handle).getroot()
                    if tree.tag not in {'mxfile', 'mxGraphModel', 'diagram'}:
                        raise ValueError('not DrawIO XML')
                    # Count original page positions, including empty pages.
                    draw_pages[f.stem] = len(tree.findall('diagram')) if tree.tag == 'mxfile' else 1
                except (ET.ParseError, ValueError, OSError) as exc:
                    errors.append(f'{f.name}: {exc}')
    Image.MAX_IMAGE_PIXELS = None  # CRC verification does not allocate a full raster.
    for f in sorted((paper/'images').glob('*.png')):
        match = IMAGE.fullmatch(f.name)
        if not match or match[1] not in draw_pages or int(match[2]) < 1:
            errors.append(f'unmatched image: {f.name}')
        elif int(match[2]) > draw_pages[match[1]]:
            warnings.append(f'export-page index exceeds XML logical-page count; historical PDF pagination can cause this: {f.name}')
        try:
            with f.open('rb', buffering=1024*1024) as handle:
                with Image.open(handle) as image:
                    if image.format != 'PNG':
                        raise ValueError('not a PNG')
                    w, h = image.size
                    if min(w,h) < 1:
                        raise ValueError('empty dimensions')
                    image.verify()
                if w*h <= decode_pixels:
                    handle.seek(0)
                    with Image.open(handle) as image:
                        image.load()
                else:
                    warnings.append(f'PNG CRC verified; pixel decoding omitted above {decode_pixels} pixels: {f.name}')
        except (OSError, ValueError, SyntaxError) as exc:
            errors.append(f'{f.name}: {exc}')
    return {'arxiv_id':paper.name, 'errors':errors, 'warnings':warnings,
            'drawio_files':len(list((paper/'drawio').glob('*.drawio'))),
            'png_images':len(list((paper/'images').glob('*.png')))}


def validate(root: Path, *, allow_links=False, decode_pixels=16_000_000, limit=None, workers=4) -> dict:
    entries = papers(root)
    with os.scandir(root) as listing:
        unexpected = [e.name for e in listing if not (ID.fullmatch(e.name) and e.is_dir())]
    if unexpected:
        raise ValueError(f'unexpected dataset-root entries: {unexpected[:10]}')
    if limit is not None:
        entries=entries[:limit]
    from concurrent.futures import ThreadPoolExecutor
    if workers < 1:raise ValueError('workers must be positive')
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i,row in enumerate(executor.map(lambda p:validate_paper(p,allow_links=allow_links,decode_pixels=decode_pixels),entries),1):
            rows.append(row)
            if i%500==0:
                print(f'validated {i}/{len(entries)} papers',flush=True)
    return {'papers':len(rows), 'drawio_files':sum(r['drawio_files'] for r in rows),
            'png_images':sum(r['png_images'] for r in rows),
            'error_count':sum(len(r['errors']) for r in rows),
            'warning_count':sum(len(r['warnings']) for r in rows),
            'issues':[r for r in rows if r['errors'] or r['warnings']]}


def require_outside(source: Path, target: Path):
    a,b=source.resolve(),target.resolve()
    if b==a or a in b.parents or b in a.parents:
        raise ValueError('source, destination and audit directories must be separate')


def export(source: Path, output: Path, *, limit=None, report: Path | None=None, skip_incomplete=False) -> dict:
    require_outside(source,output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    if report is not None:
        require_outside(output,report.parent)
    entries=papers(source)
    if limit is not None:entries=entries[:limit]
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    counts={'papers':0,'drawio_files':0,'png_images':0,'excluded':[]}
    try:
        for paper in entries:
            status=paper/'source_render_status.json'
            try:
                if status.is_file() and json.loads(status.read_text()).get('status')!='complete':
                    raise ValueError('partial rendering')
                pairs=assets(paper)
            except (ValueError,KeyError,OSError) as exc:
                if not skip_incomplete:raise
                counts['excluded'].append({'arxiv_id':paper.name,'reason':str(exc)})
                continue
            for f,name in pairs:
                dst=stage/paper.name/name;dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(f,dst,follow_symlinks=True)
                counts['drawio_files']+=name.endswith('.drawio')
                counts['png_images']+=name.endswith('.png')
            counts['papers']+=1
        if not counts['papers']:raise ValueError('no complete papers to export')
        stage.rename(output)
    finally:
        if stage.exists():shutil.rmtree(stage)
    if report is not None:
        report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(counts,indent=2)+'\n')
    return counts


def sha256(file:Path):
    h=hashlib.sha256()
    with file.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def pack(source:Path, output:Path, *, shard_bytes=2*1024**3, prefix='dataset',limit=None,workers=4,resume=False) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    require_outside(source,output)
    if shard_bytes<1 or workers<1 or not re.fullmatch(r'[A-Za-z0-9_-]+',prefix):
        raise ValueError('invalid shard size, worker count or prefix')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'archive output must be empty: {output}')
    entries=papers(source)
    if limit is not None:entries=entries[:limit]
    output.mkdir(parents=True,exist_ok=True)
    def plan_paper(paper):
        pairs=[(f,name,f.stat().st_size) for f,name in assets(paper)]
        estimated=sum(512+((length+511)//512)*512 for _f,_name,length in pairs)
        return paper.name,pairs,estimated
    groups=[];group=[];size=0
    with ThreadPoolExecutor(max_workers=workers) as planner:
        for i,item in enumerate(planner.map(plan_paper,entries),1):
            if group and size+item[2]>shard_bytes:groups.append(group);group=[];size=0
            group.append(item);size+=item[2]
            if i%5000==0:print(f'planned {i}/{len(entries)} papers',flush=True)
    if group:groups.append(group)
    class HashWriter:
        def __init__(self,file,digest=None):self.file=file;self.hash=digest or hashlib.sha256()
        def write(self,data):
            written=self.file.write(data);self.hash.update(data[:written]);return written
        def tell(self):return self.file.tell()
        def flush(self):self.file.flush()
    def write_shard(item):
        number,group=item;name=f'{prefix}-{number:05d}.tar';final=output/name
        ids=[paper for paper,_pairs,_size in group]
        expected={f'{aid}/{name}':length for aid,pairs,_size in group for _f,name,length in pairs}
        if final.exists():
            if not resume:raise FileExistsError(str(final))
            with tarfile.open(final) as archive:
                actual={m.name:m.size for m in archive if m.isfile()}
            if actual!=expected:raise ValueError(f'existing shard membership differs: {name}')
            digest=sha256(final)
            print(f'reused shard {number+1}/{len(groups)}',flush=True)
        else:
            tmp=output/('.'+name+'.partial')
            if tmp.is_symlink():raise ValueError(f'partial archive must not be a symlink: {tmp}')
            ordered=[(f,f'{aid}/{relative}',length) for aid,pairs,_size in group for f,relative,length in pairs]
            completed=0;end=0
            if resume and tmp.exists():
                available=tmp.stat().st_size
                try:
                    with tarfile.open(tmp) as previous:
                        for member in previous:
                            if completed>=len(ordered):raise ValueError(f'partial archive has extra members: {name}')
                            _f,relative,length=ordered[completed]
                            if not member.isfile() or member.name!=relative or member.size!=length:
                                raise ValueError(f'partial archive membership differs: {name}')
                            member_end=member.offset_data+((member.size+511)//512)*512
                            if member_end>available:break
                            completed+=1;end=member_end
                except tarfile.ReadError:
                    pass  # An interrupted header/payload suffix is rebuilt.
            mode='r+b' if resume and tmp.exists() else 'w+b'
            with tmp.open(mode,buffering=8*1024*1024) as file:
                digest=hashlib.sha256();remaining=end
                while remaining:
                    block=file.read(min(8*1024*1024,remaining))
                    if not block:raise ValueError(f'truncated partial prefix: {name}')
                    digest.update(block);remaining-=len(block)
                file.truncate(end);file.seek(end)
                writer=HashWriter(file,digest)
                with tarfile.open(fileobj=writer,mode='w',format=tarfile.PAX_FORMAT,dereference=True,copybufsize=1024*1024) as archive:
                    for f,relative,length in ordered[completed:]:
                        info=tarfile.TarInfo(relative);info.size=length
                        info.mtime=0;info.mode=0o644;info.uid=info.gid=0;info.uname=info.gname=''
                        with f.open('rb',buffering=1024*1024) as handle:archive.addfile(info,handle)
                writer.flush();digest=writer.hash.hexdigest()
            os.replace(tmp,final)
            if completed:print(f'resumed {completed} complete files in shard {number+1}',flush=True)
            print(f'packed shard {number+1}/{len(groups)} ({len(ids)} papers)',flush=True)
        return {'filename':name,'bytes':final.stat().st_size,'sha256':digest,'papers':ids}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records=list(executor.map(write_shard,enumerate(groups)))
    expected_names={r['filename'] for r in records}
    if {f.name for f in output.glob('*.tar')}!=expected_names:raise ValueError('unexpected archive files in output')
    result={'papers':len(entries),'shards':records,'source_bytes_resolved':True}
    (output/'checksums.sha256').write_text(''.join(f"{r['sha256']}  {r['filename']}\n" for r in records))
    (output/'pack_report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def check_archives(output:Path) -> dict:
    seen_papers=set();files=0
    report=json.loads((output/'pack_report.json').read_text())
    for row in report['shards']:
        file=output/row['filename']
        if sha256(file)!=row['sha256']:raise ValueError(f'checksum mismatch: {file.name}')
        shard_papers=set()
        with tarfile.open(file) as t:
            seen=set()
            for item in t:
                p=PurePosixPath(item.name)
                if not item.isfile() or p.is_absolute() or '..' in p.parts or len(p.parts) not in (2,3):
                    raise ValueError(f'non-portable member: {item.name}')
                if not ID.fullmatch(p.parts[0]):raise ValueError('invalid paper ID')
                valid=(len(p.parts)==2 and p.parts[1]=='paper.md') or (len(p.parts)==3 and ((p.parts[1]=='drawio' and p.suffix=='.drawio') or (p.parts[1]=='images' and p.suffix=='.png')))
                if not valid or item.name in seen:raise ValueError(f'unexpected/duplicate member: {item.name}')
                seen.add(item.name);shard_papers.add(p.parts[0]);files+=1
        if shard_papers!=set(row['papers']) or seen_papers & shard_papers:
            raise ValueError('paper membership differs from report or spans shards')
        seen_papers.update(shard_papers)
    if len(seen_papers)!=report['papers']:raise ValueError('paper count mismatch')
    return {'papers':len(seen_papers),'shards':len(report['shards']),'files':files,'checksums_verified':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['export','validate','pack','check-archives'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--limit',type=int)
    p.add_argument('--allow-links',action='store_true')
    p.add_argument('--skip-incomplete',action='store_true')
    p.add_argument('--shard-gib',type=float,default=2)
    p.add_argument('--prefix',default='dataset')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--decode-pixels',type=int,default=16_000_000)
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('limit must be positive')
    if args.action in ['export','pack'] and args.output is None:p.error('--output is required')
    if args.report is not None:require_outside(args.source,args.report.parent)
    if args.action=='export':result=export(args.source,args.output,limit=args.limit,report=args.report,skip_incomplete=args.skip_incomplete)
    elif args.action=='validate':result=validate(args.source,allow_links=args.allow_links,limit=args.limit,decode_pixels=args.decode_pixels,workers=args.workers)
    elif args.action=='pack':result=pack(args.source,args.output,shard_bytes=int(args.shard_gib*1024**3),prefix=args.prefix,limit=args.limit,workers=args.workers,resume=args.resume)
    else:result=check_archives(args.source)
    if args.report is not None:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'issues','shards'}},sort_keys=True))
    if result.get('error_count',0):raise SystemExit(2)

if __name__=='__main__':main()
