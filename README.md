# FigAgentBenchmark

**Paper:** [Automatic Method Illustration Generation for AI Scientific Papers via Drawing Middleware Creation, Evolution, and Orchestration](https://arxiv.org/abs/2603.29590)

| Collection | Google Drive | Baidu Netdisk |
|---|---|---|
| FigAgent-5K | [Download](https://drive.google.com/drive/folders/1UJHgOX6-kbPtGBjqMMbB587NQNo1w7H4) | [Download](https://pan.baidu.com/s/1tomjm8H9aMczEc-K2nUiOw?pwd=rxv9) (code: `rxv9`) |
| FigAgent-Corpus | [Download](https://drive.google.com/drive/folders/1vKXFKnVKzLLUC1PeO-1mfhq1uhouPNHk) | [Download](https://pan.baidu.com/s/1tomjm8H9aMczEc-K2nUiOw?pwd=rxv9) (code: `rxv9`) |

This repository provides **FigAgent-5K** and **FigAgent-Corpus**, with dataset descriptions, download resources, and collection code.

## Overview

FigAgent-5K and FigAgent-Corpus pair scientific paper Markdown with editable DrawIO documents and rendered PNG images. These representations support scientific figure generation, editing, understanding, and reuse of drawing components.

## Dataset

| Collection | Papers | DrawIO documents | Rendered PNG pages |
|---|---:|---:|---:|
| **FigAgent-5K** | 5,000 | 7,170 | 7,681 |
| **FigAgent-Corpus** | 48,500 | 153,143 | 207,209 |

**FigAgent-5K** contains 5,000 curated papers selected through deduplication, structural screening, and semantic filtering. **FigAgent-Corpus** provides a larger collection of editable diagrams from scientific papers.

Each paper has a folder named by its arXiv ID:

```text
<arxiv_id>/
├── paper.md
├── drawio/
│   └── diagram.drawio
└── images/
    ├── diagram_p001.png
    └── diagram_p002.png
```

PNG filenames match the DrawIO stem and include a rendered page index. A DrawIO document can have multiple pages.

## Data Collection

Paper sources are collected from arXiv. Editable diagrams are recovered and rendered, then paired with paper text. Completeness checks produce FigAgent-Corpus, while additional deduplication and quality screening produce FigAgent-5K.

## Build Your Own Dataset

Use the stage runner to collect and prepare a dataset for your chosen paper range and output directories.

### Set up

On Linux, install DrawIO Desktop, Xvfb, DBus, and Pandoc. Install the Python dependencies:

```bash
git clone https://github.com/zhuolingli/FigAgentBenchmark.git
cd FigAgentBenchmark
pip install -e '.[render]'
export DRAWIO_DESKTOP_BIN=/path/to/drawio-desktop
cp configs/collection.example.json configs/my_collection.json
```

In `configs/my_collection.json`, set `collection.from_month` and `collection.to_month` in `YYMM` format, from newest to oldest. Choose your working and output directories under `paths`.

### Configure AWS access

The default collection workflow downloads source archives from `s3://arxiv/src/` in `us-east-1`. You need an AWS account with billing enabled and credentials with access to this bucket. arXiv uses **Requester Pays**: requests and downloads are charged to your AWS account. See [arXiv bulk access](https://info.arxiv.org/help/bulk_data_s3.html).

Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then configure a named profile using your IAM access key and secret key:

```bash
aws configure --profile figagent
aws configure set region us-east-1 --profile figagent
export AWS_PROFILE=figagent
aws sts get-caller-identity
```

If your organisation uses IAM Identity Center, use [`aws configure sso --profile figagent`](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html) and `aws sso login --profile figagent` instead, then export the same `AWS_PROFILE`.

Attach the following read permissions to the IAM user or role used by your profile:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::arxiv",
      "Condition": {"StringLike": {"s3:prefix": ["src/*"]}}
    },
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::arxiv/src/*"
    }
  ]
}
```

Check access by listing one source archive:

```bash
aws s3api list-objects-v2 --bucket arxiv --prefix src/arXiv_src_2410_ \
    --request-payer requester --region us-east-1 --max-keys 1 \
    --query 'Contents[].Key'
```

The collection scripts use the standard AWS credential chain and supply the Requester Pays flag automatically. Keep `AWS_PROFILE` set in the shell running the pipeline. On EC2, an instance IAM role with the same permissions can supply credentials directly. Store credentials in your AWS profile or environment, outside the repository.

### Collect and prepare

```bash
python scripts/run_pipeline.py --stage collect --config configs/my_collection.json --dry-run
python scripts/run_pipeline.py --stage collect --config configs/my_collection.json
python scripts/run_pipeline.py --stage recover --config configs/my_collection.json
python scripts/run_pipeline.py --stage render --config configs/my_collection.json
python scripts/run_pipeline.py --stage text --config configs/my_collection.json
python scripts/run_pipeline.py --stage export --dataset corpus --config configs/my_collection.json
python scripts/run_pipeline.py --stage validate --dataset corpus --config configs/my_collection.json
```

Start with the default small run. Add `--full` to `collect`, `recover`, `render`, and `text` to process the configured range. Use a new export directory for each collection.

The result is a dataset of paper folders containing `paper.md`, `drawio/`, and `images/`.

### Optional quality screening

For a curated collection, install the quality dependencies and use a CUDA-capable GPU:

```bash
pip install -e '.[quality]'
python scripts/run_pipeline.py --stage curate --config configs/my_collection.json
python scripts/run_pipeline.py --stage export --dataset selected --config configs/my_collection.json
python scripts/run_pipeline.py --stage validate --dataset selected --config configs/my_collection.json
```

The `selected` output uses your collected papers; its destination is configured by `paths.release_selected`.

## Download and Usage

### Installation

Use Python 3.11 or newer.

```bash
pip install -e .
```

### Download and read the data

Download every TAR shard for the selected collection from either mirror listed above. Each download folder contains only the data shards.

| Collection | TAR shards | Archive size |
|---|---:|---:|
| FigAgent-5K | 4 | 8.15 GB (7.59 GiB) |
| FigAgent-Corpus | 63 | 133.95 GB (124.75 GiB) |

Extract all shards into the same collection directory. Each paper stays within one shard and contains `paper.md`, `drawio/*.drawio`, and `images/*.png`.

```bash
mkdir -p data/FigAgent-5K
for shard in downloads/figagent-5k-v1-*.tar; do
    tar -xf "$shard" -C data/FigAgent-5K
done
python examples/read_dataset.py --dataset data/FigAgent-5K --limit 1
```

The reader enumerates paper folders and matches PNG pages to DrawIO documents by filename.

## License and Citation

Collection and curation code is released under the **MIT license**. Paper texts and figure assets retain their original source licenses and authors' rights.

If you use the datasets or collection code, please cite:

```bibtex
@article{li2026figagent,
  title={Automatic Method Illustration Generation for AI Scientific Papers via Drawing Middleware Creation, Evolution, and Orchestration},
  author={Li, Zhuoling and Zhang, Jiarui and Hu, Ping and Kuen, Jason and Gu, Jiuxiang and Rahmani, Hossein and Liu, Jun},
  journal={arXiv preprint arXiv:2603.29590},
  year={2026}
}
```
