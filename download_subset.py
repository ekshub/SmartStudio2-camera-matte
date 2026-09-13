"""Download a small Matting Human evaluation subset via Kaggle Bearer token."""
from pathlib import Path
import os, time, requests
DATASET="laurentmih/aisegmentcom-matting-human-datasets"; BASE="https://www.kaggle.com/api/v1"; OUT=Path("datasets/matting_human_subset"); N=int(os.getenv("SMARTSTUDIO_SAMPLES","100")); TOKEN=os.environ.get("KAGGLE_API_TOKEN")
if not TOKEN: raise SystemExit("Set KAGGLE_API_TOKEN before running this script")
headers={"Authorization":f"Bearer {TOKEN}"}
def get_json(url,params):
    for i in range(8):
        try:
            r=requests.get(url,headers=headers,params=params,timeout=90); r.raise_for_status(); return r.json()
        except Exception as exc:
            if i==7: raise
            print(f"retry: {exc}",flush=True); time.sleep(2+i*2)
names=[]; token=None
while True:
    params={"pageSize":200};
    if token: params["pageToken"]=token
    data=get_json(f"{BASE}/datasets/list/{DATASET}",params); names.extend(x["name"] for x in data.get("datasetFiles",[]) if x.get("name")); print(f"listed {len(names)} files",flush=True)
    token=data.get("nextPageToken") or data.get("nextPageTokenNullable")
    if not token: break
images=[n for n in names if n.startswith("clip_img/") and n.lower().endswith((".jpg",".jpeg",".png"))]; alphas=[n for n in names if n.startswith("matting/") and n.lower().endswith(".png")]; by_stem={Path(n).stem:n for n in alphas}; pairs=[(im,by_stem[Path(im).stem]) for im in images if Path(im).stem in by_stem][:N]; print(f"matched pairs: {len(pairs)}",flush=True)
def download(name,folder):
    out=folder/Path(name).name
    if out.exists() and out.stat().st_size>0:return out
    with requests.get(f"{BASE}/datasets/download/{DATASET}",headers=headers,params={"filename":name},stream=True,timeout=120) as r:
        r.raise_for_status(); tmp=out.with_suffix(out.suffix+".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:f.write(chunk)
        tmp.replace(out)
    return out
(OUT/"clip_img").mkdir(parents=True,exist_ok=True); (OUT/"matting").mkdir(parents=True,exist_ok=True)
for i,(im,alpha) in enumerate(pairs,1): print(f"[{i}/{len(pairs)}] {Path(im).name}",flush=True) or (download(im,OUT/"clip_img"),download(alpha,OUT/"matting"))
print("done")
