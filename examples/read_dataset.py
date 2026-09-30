#!/usr/bin/env python3
"""Read data-only paper folders using the filename pairing convention."""
import argparse,json,re
from pathlib import Path

def read_paper(paper):
    pairs=[]
    for image in sorted((paper/'images').glob('*.png')):
        match=re.fullmatch(r'(.+)_p(\d{3,})\.png',image.name)
        if not match:raise ValueError(f'bad image name: {image.name}')
        draw=paper/'drawio'/(match[1]+'.drawio')
        if not draw.is_file():raise ValueError(f'missing DrawIO for {image}')
        pairs.append({'drawio':str(draw),'image':str(image),'page_number':int(match[2])})
    return {'arxiv_id':paper.name,'text':(paper/'paper.md').read_text(),'figures':pairs}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--limit',type=int,default=1);a=p.parse_args()
    for paper in sorted(d for d in a.dataset.iterdir() if d.is_dir())[:a.limit]:
        row=read_paper(paper);row['text_characters']=len(row.pop('text'));print(json.dumps(row,indent=2))
if __name__=='__main__':main()
