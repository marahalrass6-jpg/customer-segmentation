from pathlib import Path
from zipfile import ZipFile
import io
import requests
URL='https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip'
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data'/'raw'; RAW.mkdir(parents=True,exist_ok=True)
r=requests.get(URL,timeout=180); r.raise_for_status()
with ZipFile(io.BytesIO(r.content)) as z: z.extractall(RAW)
print(f'Extracted {len(r.content):,} bytes to {RAW}')
for p in sorted(RAW.rglob('*.xlsx')): print(p)
