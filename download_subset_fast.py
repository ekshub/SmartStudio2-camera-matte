from pathlib import Path
import os, requests, time
TOKEN=os.environ['KAGGLE_API_TOKEN']; H={'Authorization':'Bearer '+TOKEN}; D='laurentmih/aisegmentcom-matting-human-datasets'; BASE='https://www.kaggle.com/api/v1'; out=Path('datasets/matting_human_subset'); out.mkdir(parents=True,exist_ok=True)
token=None; imgs=[]; alphas={}; page=0
while len(imgs)<100 or len([x for x in imgs if Path(x).stem in alphas])<100:
 p={'pageSize':200};
 if token:p['pageToken']=token
 for k in range(6):
  try:r=requests.get(f'{BASE}/datasets/list/{D}',headers=H,params=p,timeout=60); r.raise_for_status(); j=r.json(); break
  except Exception as e: print('retry',e,flush=True); time.sleep(2)
 names=[x['name'] for x in j.get('datasetFiles',[]) if x.get('name')]; page+=1
 for n in names:
  if n.startswith('clip_img/') and len(imgs)<100: imgs.append(n)
  if n.startswith('matting/') and n.endswith('.png'): alphas[Path(n).stem]=n
 print(f'page {page}, images {len(imgs)}, matched {len([x for x in imgs if Path(x).stem in alphas])}',flush=True)
 token=j.get('nextPageToken') or j.get('nextPageTokenNullable')
 if not token:break
pairs=[(x,alphas[Path(x).stem]) for x in imgs if Path(x).stem in alphas]
def dl(name,folder):
 folder.mkdir(parents=True,exist_ok=True); f=folder/Path(name).name
 if f.exists() and f.stat().st_size:return
 for attempt in range(10):
  try:
   with requests.get(f'{BASE}/datasets/download/{D}',headers=H,params={'filename':name},stream=True,timeout=120) as r:
    r.raise_for_status()
    with f.open('wb') as w:
     for c in r.iter_content(1024*1024):
      if c:w.write(c)
   return
  except Exception as e:
   try: f.unlink()
   except OSError: pass
   print(f'retry file {Path(name).name}: {e}',flush=True); time.sleep(2+attempt)
for i,(im,al) in enumerate(pairs,1):print(f'download {i}/{len(pairs)}',flush=True);dl(im,out/'clip_img');dl(al,out/'matting')
print('DONE',len(pairs),flush=True)
