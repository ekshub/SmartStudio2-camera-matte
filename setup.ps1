param([switch]$CreateEnv)
if ($CreateEnv) { conda env create -f environment.yml }
conda run -n smartstudio python -m pip install -r requirements.txt
$model='models/rvm_mobilenetv3.pth'
if (!(Test-Path $model)) { Invoke-WebRequest 'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth' -OutFile $model }
Write-Host "SmartStudio ready. Run: conda run -n smartstudio python web_server.py"
