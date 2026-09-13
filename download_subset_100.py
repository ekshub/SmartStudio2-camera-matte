from pathlib import Path
import os, requests, time
t=os.environ['KAGGLE_API_TOKEN']; H={'Authorization':'Bearer '+t}; D='laurentmih/aisegmentcom-matting-human-datasets'; B='https://www.kaggle.com/api/v1'; out=Path('datasets/matting_human_subset'); out.mkdir(parents=True,exist_ok=True)
j=requests.get(f'{B}/datasets/list/{D}',headers=H,params={'pageSize':200},timeout=90).json()
imgs=[x['name'] for x in j['datasetFiles'] if x['name'].startswith('clip_img/')][:100]
def alpha_path(im):
    p=Path(im); parts=im.split('/'); parts[0]='matting'; parts[2]=parts[2].replace('clip_','matting_'); return '/'.join(parts[:-1]+[p.stem+'.png'])
def dl(name,folder):
    folder.mkdir(parents=True,exist_ok=True); f=folder/Path(name).name
    if f.exists() and f.stat().st_size:return
    for k in range(10):
        try:
            with requests.get(f'{B}/datasets/download/{D}',headers=H,params={'filename':name},stream=True,timeout=120) as r:
                r.raise_for_status()
                with f.open('wb') as w:
                    for c in r.iter_content(1024*1024):
                        if c:w.write(c)
            return
        except Exception as e:
            try:f.unlink()
            except OSError:pass
            print('retry',Path(name).name,e,flush=True); time.sleep(2+k)
for i,im in enumerate(imgs,1):
    al=alpha_path(im); print(f'[{i}/100] {Path(im).name}',flush=True); dl(im,out/'clip_img'); dl(al,out/'matting')
