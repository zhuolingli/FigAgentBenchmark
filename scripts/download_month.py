#!/usr/bin/env python3
"""Download and conservatively scan arXiv's official requester-pays S3 source TARs."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, tarfile, tempfile
from pathlib import Path
import boto3
from drawio_cloud.detector import inspect_source_package
from drawio_cloud.months import descending_months


def collect(source:Path, *, from_month='2410', to_month='2410', max_objects=1, full=False):
    if max_objects<0 or (max_objects==0 and not full):
        raise ValueError('an unlimited scan requires --full')
    source.mkdir(parents=True,exist_ok=True)
    cache=source/'.archives';cache.mkdir(exist_ok=True)
    state=source/'.state';state.mkdir(exist_ok=True)
    s3=boto3.client('s3',region_name='us-east-1')
    existing=source/'selected_sources.jsonl'
    rows={r['arxiv_id']:r for r in (json.loads(s) for s in existing.read_text().splitlines())} if existing.exists() else {}
    processed=0
    for month in descending_months(from_month,to_month):
        paginator=s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket='arxiv',Prefix=f'src/arXiv_src_{month}_',RequestPayer='requester'):
            for obj in page.get('Contents',[]):
                key=obj['Key']
                if not key.endswith('.tar'):continue
                if max_objects and processed>=max_objects:return len(rows)
                processed+=1
                done=state/(Path(key).name+'.done')
                if done.exists():continue
                if shutil.disk_usage(source).free<max(obj['Size']*3,2*1024**3):
                    raise RuntimeError('insufficient working space for source archive')
                local=cache/Path(key).name;partial=local.with_suffix('.tar.partial')
                if not local.exists() or local.stat().st_size!=obj['Size']:
                    s3.download_file('arxiv',key,str(partial),ExtraArgs={'RequestPayer':'requester'})
                    if partial.stat().st_size!=obj['Size']:raise ValueError('source archive size mismatch')
                    partial.replace(local)
                with tarfile.open(local,'r:*') as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith('.gz'):continue
                        aid=Path(member.name).stem
                        if not aid.replace('.','').isdigit():continue
                        with tempfile.NamedTemporaryFile(dir=cache,suffix='.gz') as temp:
                            stream=archive.extractfile(member)
                            if stream is None:continue
                            with stream:shutil.copyfileobj(stream,temp)
                            temp.flush()
                            result=inspect_source_package(temp.name)
                            if not result.matched:continue
                            payload=Path(temp.name);destination=source/month/(aid+'.gz')
                            destination.parent.mkdir(parents=True,exist_ok=True)
                            shutil.copyfile(payload,destination)
                            digest=hashlib.sha256(payload.read_bytes()).hexdigest()
                            rows[aid]={'arxiv_id':aid,'month':month,'source_tar':Path(key).name,
                                       'source_key':key,'source_member':member.name,'size':member.size,
                                       'sha256':digest,'reasons':list(result.reasons)}
                temporary=existing.with_suffix('.jsonl.tmp')
                temporary.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in sorted(rows.values(),key=lambda r:r['arxiv_id'])))
                os.replace(temporary,existing);done.write_text('complete\n');local.unlink()
                print(f'scanned {key}; retained {len(rows)} papers',flush=True)
    return len(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest',type=Path,required=True)
    p.add_argument('--from-month',default='2410');p.add_argument('--to-month',default='2410')
    p.add_argument('--max-source-objects',type=int,default=1);p.add_argument('--full',action='store_true')
    a=p.parse_args();print('retained',collect(a.dest,from_month=a.from_month,to_month=a.to_month,max_objects=a.max_source_objects,full=a.full))
if __name__=='__main__':main()
