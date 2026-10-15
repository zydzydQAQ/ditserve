#!/usr/bin/env python3
"""Record real local Docker image sizes and build-layer sizes; never estimate missing images."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('images', nargs='*', default=['ditserve:a100-runtime', 'ditserve:a100-longlive'])
args = parser.parse_args()
rows = []
for image in args.images:
    result = subprocess.run(['docker', 'image', 'inspect', image], text=True, capture_output=True)
    if result.returncode:
        rows.append({'image': image, 'status': 'unverified', 'error': result.stderr.strip(), 'size_bytes': None})
        continue
    info = json.loads(result.stdout)[0]
    history = subprocess.run(['docker', 'history', '--no-trunc', '--human=false', '--format', '{{json .}}', image],
                             text=True, capture_output=True, check=True)
    rows.append({'image': image, 'status': 'built', 'id': info['Id'], 'size_bytes': info['Size'],
                 'created': info['Created'], 'platform': info['Os'] + '/' + info['Architecture'],
                 'layer_diff_ids': info['RootFS']['Layers'],
                 'history': [json.loads(line) for line in history.stdout.splitlines()]})
report = {'schema_version': 1, 'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'images': rows, 'gpu_validation': 'not_run; requires target 8 x A100 host'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(args.output)
