#!/usr/bin/env python3
"""Bridge internal paper bundles to the preserved index/dedup/SigLIP pipeline."""
from __future__ import annotations
import argparse,json,shutil,urllib.request,urllib.parse,time,xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def categories(dataset:Path, output:Path, *, delay=3.0):
    """Obtain only title/category metadata; figure source packages come from S3."""
    cached=json.loads(output.read_text()) if output.exists() else {}
    entries=read_rows(dataset/'manifest.jsonl');todo=[r['arxiv_id'] for r in entries if r['arxiv_id'] not in cached]
    ns={'a':'http://www.w3.org/2005/Atom'}
    for start in range(0,len(todo),50):
        ids=todo[start:start+50]
        url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'id_list':','.join(ids),'max_results':50})
        request=urllib.request.Request(url,headers={'User-Agent':'FigAgentBenchmark/0.1 (research metadata lookup)'})
        with urllib.request.urlopen(request,timeout=60) as response:tree=ET.fromstring(response.read())
        for entry in tree.findall('a:entry',ns):
            aid=(entry.findtext('a:id',default='',namespaces=ns).rsplit('/',1)[-1]).split('v')[0]
            if aid in ids:cached[aid]={'categories':[c.attrib['term'] for c in entry.findall('a:category',ns)],'title':' '.join(entry.findtext('a:title',default='',namespaces=ns).split())}
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(cached,sort_keys=True,indent=2)+'\n');tmp.replace(output)
        if start+50<len(todo):time.sleep(delay)
    return cached


def index(dataset:Path, output:Path, *, categories_file:Path|None=None, cs_only=True, workers=1):
    from build_figagent_score_index import index_paper
    metadata=json.loads(categories_file.read_text()) if categories_file and categories_file.exists() else {}
    entries=[r for r in read_rows(dataset/'manifest.jsonl') if (dataset/'papers'/r['arxiv_id']/'paper.md').is_file() and (dataset/'papers'/r['arxiv_id']/'paper.md').read_text().strip()]
    output.parent.mkdir(parents=True,exist_ok=True)
    unknown=0;count=0;tmp=output.with_suffix('.jsonl.tmp')
    with tmp.open('w') as handle,ThreadPoolExecutor(max_workers=workers) as executor:
        for entry,rows in zip(entries,executor.map(index_paper,[(dataset,r) for r in entries])):
            aid=entry['arxiv_id'];cats=metadata.get(aid,{}).get('categories',[])
            if not cats:
                f=dataset/'papers'/aid/'source_record.json'
                if f.exists():cats=json.loads(f.read_text()).get('categories') or []
            if isinstance(cats,str):cats=cats.split()
            if cs_only and not cats:unknown+=1
            eligible=any(c.startswith('cs.') for c in cats) if cs_only else True
            for row in rows:
                row.update(is_cs=eligible,is_main=False,old_quality_pass=False,old_quality_score=None)
                handle.write(json.dumps(row,sort_keys=True)+'\n');count+=1
    if cs_only and unknown:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'missing category metadata for {unknown} papers; run text metadata retrieval before CS-only curation')
    tmp.replace(output);print(f'indexed {count} images',flush=True)


def _build_selected(selected:Path, corpus:Path, output:Path, *, threshold=0.70263536, expected_papers=None):
    rows=[r for r in read_rows(selected) if r['structure_score']>=threshold]
    groups={}
    for r in rows:groups.setdefault(r['arxiv_id'],[]).append(r)
    if not groups:raise ValueError('no figures passed the structure threshold')
    if expected_papers is not None and len(groups)!=expected_papers:
        raise ValueError(f'expected {expected_papers} papers, found {len(groups)}; frozen inputs are required for exact membership')
    if output.exists():raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    manifest=[]
    for aid,figures in sorted(groups.items()):
        paper=output/'papers'/aid
        (paper/'drawio').mkdir(parents=True);(paper/'images').mkdir()
        body=corpus/'papers'/aid/'paper.md'
        if not body.is_file() or not body.read_text().strip():raise ValueError(f'missing text for {aid}')
        (paper/'paper.md').symlink_to(body.resolve())
        mapping=[]
        for row in figures:
            draw=Path(row['source_drawio']);image=Path(row['png'])
            dname=draw.name;iname=f'{draw.stem}_p{int(row["page_number"]):03d}.png'
            for source,target in [(draw,paper/'drawio'/dname),(image,paper/'images'/iname)]:
                if target.exists():
                    if target.resolve()!=source.resolve():raise ValueError(f'conflicting asset: {target}')
                else:target.symlink_to(source.resolve())
            mapping.append({'drawio_path':'drawio/'+dname,'image_path':'images/'+iname,'page_number':int(row['page_number'])})
        (paper/'metadata.json').write_text(json.dumps({'paper':{'arxiv_id':aid,'title':None},'figures':mapping},indent=2)+'\n')
        (paper/'text_status.json').write_text(json.dumps({'arxiv_id':aid,'status':'fetched'})+'\n')
        manifest.append({'arxiv_id':aid,'folder':'papers/'+aid,'paper_markdown_path':f'papers/{aid}/paper.md','metadata_path':f'papers/{aid}/metadata.json','text_status_path':f'papers/{aid}/text_status.json'})
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    print(f'selected papers={len(groups)} images={len(rows)}',flush=True)


def build_selected(selected:Path, corpus:Path, output:Path, **kwargs):
    import tempfile
    if output.exists():raise FileExistsError(f'refusing to overwrite {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.selected-',dir=output.parent))
    try:
        _build_selected(selected,corpus,stage/'dataset',**kwargs)
        (stage/'dataset').rename(output)
    finally:
        shutil.rmtree(stage,ignore_errors=True)
