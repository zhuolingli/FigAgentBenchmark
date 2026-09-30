#!/usr/bin/env python3
"""A thin, explicit stage runner for the existing collection algorithms."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parent
REPO=SCRIPTS.parent
STAGES=('collect','recover','render','text','curate','export','validate')


def commands(config:dict, stage:str, *, full=False, dataset='corpus', cloud_action='provision'):
    dataset = '5k' if dataset == 'selected' else dataset
    base=Path(config.get('base_dir','.')).resolve()
    paths=config['paths']
    selected_release_key = 'release_selected' if paths.get('release_selected') else 'release_5k'
    path=lambda key:str((base/paths[key]).resolve())
    work=path('work');source=path('source');recovery=path('recovery');corpus=path('corpus');selected=path('selected')
    scratch=path('scratch') if paths.get('scratch') else str(Path(tempfile.gettempdir())/f'figagent-scratch-{os.getuid()}')
    py=sys.executable
    script=lambda n:str(SCRIPTS/n)
    workers=str(config.get('workers',1))
    limit=[] if full else ['--limit','1']
    quality=str(Path(work)/'QUALITY_SELECTION_V2')
    env={**os.environ,'DRAWIO_QUALITY_ROOT':work,'DRAWIO_QUALITY_OUTPUT':quality,
         'DRAWIO_BIN':config.get('drawio_bin','drawio')}
    jobs=[]
    if stage=='collect':
        c=config.get('collection',{});backend=c.get('backend','local')
        cap='0' if full else '1'
        if backend=='local':
            jobs=[[py,script('download_month.py'),'--dest',source,'--from-month',c.get('from_month','2410'),'--to-month',c.get('to_month','2410'),'--max-source-objects',cap]]
        elif backend=='cloud':
            jobs=[[py,str(REPO/'tools/cloudctl.py'),cloud_action,'--from-month',c.get('from_month','2410'),'--to-month',c.get('to_month','2410'),'--workers',str(c.get('cloud_workers',1) if full else 1),'--max-source-objects',cap,'--max-hours',str(c.get('max_hours',168))]]
        else:raise ValueError('collection.backend must be local or cloud')
        if full:jobs[0].append('--full')
    elif stage=='recover':
        jobs=[[py,script('recover_source_packages.py'),'--manifest',str(Path(source)/'selected_sources.jsonl'),'--source-root',source,'--output',recovery,'--scratch',str(Path(scratch)/'recover'),'--workers',workers,*limit,*(['--retry-failed'] if config.get('recovery',{}).get('retry_failed',False) else [])]]
    elif stage=='render':
        cap=[] if full else ['--limit-papers','1']
        jobs=[[py,script('render_recovery_source2.py'),'--root',recovery,'--scratch',str(Path(scratch)/'render'),'--workers',workers,*cap],
              [py,script('import_recovery_source2.py'),'--source',recovery,'--output',corpus,'--workers',workers,*cap]]
    elif stage=='text':
        t=config.get('text',{});provider=t.get('provider','markdown')
        if provider != 'markdown':raise ValueError('text.provider must be markdown')
        jobs=[[py,script('fetch_alphaxiv_markdown.py'),'--dataset',corpus,*limit]]
        if t.get('source_fallback',True):jobs.append([py,script('fetch_arxiv_source_fallback.py'),'--dataset',corpus,*limit,*(['--lightweight'] if t.get('lightweight_fallback',False) else [])])
        if t.get('fetch_categories',True):
            jobs.append([py,'-c','import sys; from pathlib import Path;sys.path.insert(0,sys.argv[1]);from curate_dataset import categories;categories(Path(sys.argv[2]),Path(sys.argv[3]))',str(SCRIPTS),corpus,str(Path(work)/'categories.json')])
    elif stage=='curate':
        c=config.get('quality',{});legacy=c.get('legacy_root')
        if legacy:
            env['DRAWIO_QUALITY_DATASET']=str((base/legacy).resolve())
            jobs=[[py,script('quality_select_v2.py'),'--stage','all','--workers',workers]]
        else:
            jobs=[[py,'-c','import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from curate_dataset import index;index(Path(sys.argv[2]),Path(sys.argv[3]),categories_file=Path(sys.argv[4]),cs_only=sys.argv[5]=="true",workers=int(sys.argv[6]))',str(SCRIPTS),corpus,str(Path(quality)/'all_pages.jsonl'),str(Path(work)/'categories.json'),str(c.get('cs_only',True)).lower(),workers],
                  [py,script('quality_select_v2.py'),'--stage','dedup']]
        jobs.extend([[py,script('prepare_quality_candidates_v2.py'),'--input',str(Path(quality)/'unique_pages.jsonl'),'--retain-fraction',str(c.get('retain_fraction',0.6)),*(['--threshold-config',str((base/c['frozen_policy']).resolve())] if c.get('frozen_policy') else [])],
          [py,script('siglip_filter_v2.py'),'--image-views','dual','--gpu',str(c.get('gpu',0))],
          [py,script('finalize_quality_selection_v2.py'),'--margin-threshold',str(c.get('margin_threshold',2.75)),'--top-n','0'],
          [py,'-c','import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from curate_dataset import build_selected;build_selected(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]),threshold=float(sys.argv[5]),expected_papers=None if sys.argv[6]=="null" else int(sys.argv[6]))',str(SCRIPTS),str(Path(quality)/'selected.jsonl'),corpus,selected,str(c.get('structure_threshold',0.70263536)),'null' if c.get('expected_papers') in (None, 'null') else str(c['expected_papers'])]])
    elif stage=='export':
        inp=path('export_source') if paths.get('export_source') else (selected if dataset=='5k' else corpus)
        out=path(selected_release_key if dataset=='5k' else 'release_corpus')
        audit=str(Path(work)/'audit'/f'export-{dataset}.json')
        jobs=[[py,script('release_assets.py'),'export','--source',inp,'--output',out,'--report',audit,'--skip-incomplete']]
    elif stage=='validate':
        inp=path(selected_release_key if dataset=='5k' else 'release_corpus')
        jobs=[[py,script('release_assets.py'),'validate','--source',inp,'--report',str(Path(work)/'audit'/f'validate-{dataset}.json')]]
    return jobs,env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dataset',choices=['corpus','selected','5k'],default='corpus')
    p.add_argument('--cloud-action',choices=['provision','start','status','stop'],default='provision')
    p.add_argument('--full',action='store_true',help='Explicitly expand smoke-test collection/recovery/rendering limits')
    p.add_argument('--dry-run',action='store_true',help='Print commands without running or writing any state')
    a=p.parse_args();config=json.loads(a.config.read_text())
    config['base_dir']=str((a.config.parent/config.get('base_dir','..')).resolve())
    jobs,env=commands(config,a.stage,full=a.full,dataset=a.dataset,cloud_action=a.cloud_action)
    for command in jobs:
        print(json.dumps(command),flush=True)
        if not a.dry_run:
            result=subprocess.run(command,env=env,cwd=REPO,check=False)
            if result.returncode == 2 and Path(command[1]).name == 'render_recovery_source2.py':
                print('Some papers failed rendering; importing only complete papers.',file=sys.stderr)
            elif result.returncode:
                raise subprocess.CalledProcessError(result.returncode,command)
if __name__=='__main__':main()
