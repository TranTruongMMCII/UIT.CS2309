#!/usr/bin/env bash
set -euo pipefail
mkdir -p /kaggle/working/restored_checkpoints
curl -L -o /kaggle/working/restored_checkpoints/phase1_model_5_step5fresh_20260621_122349_64f0adf.pth https://github.com/TranTruongMMCII/UIT.CS2309/releases/download/step5-step5fresh_20260621_122349_64f0adf/phase1_model_5_step5fresh_20260621_122349_64f0adf.pth
echo "Downloaded to /kaggle/working/restored_checkpoints/phase1_model_5_step5fresh_20260621_122349_64f0adf.pth"
python - <<'PYVERIFY'
import hashlib
from pathlib import Path
path = Path('/kaggle/working/restored_checkpoints/phase1_model_5_step5fresh_20260621_122349_64f0adf.pth')
expected = '57303284b98a23b9c21c8f44fd4ac7ae13e45b71f30f490e0de5c247e1fd1c70'
h = hashlib.sha256(path.read_bytes()).hexdigest()
print('sha256:', h)
assert h == expected, 'SHA256 mismatch'
PYVERIFY
