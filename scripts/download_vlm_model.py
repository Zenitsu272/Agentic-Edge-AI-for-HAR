"""Download pinned local VLM weights. This downloads data, not remote model code."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

REPOSITORY = 'unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit'
REVISION = 'f9adca15cefb612e7f03a1c8a5fddb920b1786f3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'.venv/models/Qwen2.5-VL-3B-Instruct-bnb-4bit'))
    args = parser.parse_args()
    print('Downloading about 2.47 GB of local VLM weights. Existing downloads are resumed.',flush=True)
    folder = snapshot_download(REPOSITORY,revision=REVISION,local_dir=args.output,
                               allow_patterns=['*.json','*.safetensors','merges.txt','tokenizer.model','LICENSE*','README.md'],
                               max_workers=4)
    print('Local model ready:',folder)
