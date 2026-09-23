$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_HOME = "C:\phon\hf"
$py = "C:\phon\venv\Scripts\python.exe"
New-Item -ItemType Directory -Force C:\phon\logs | Out-Null
Start-Process $py -ArgumentList "C:\phon\code\keepawake.py", $PID -WindowStyle Hidden
function Step($name, $argline) {
  "$(Get-Date -Format 'HH:mm') START $name" | Add-Content C:\phon\logs\runner.log
  $p = Start-Process $py -ArgumentList $argline -WorkingDirectory C:\phon\code -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "C:\phon\logs\$name.log" -RedirectStandardError "C:\phon\logs\$name.err"
  "$(Get-Date -Format 'HH:mm') END   $name exit $($p.ExitCode)" | Add-Content C:\phon\logs\runner.log
}
# 1. wait for the paper-reciter prep launched separately
while (-not (Select-String -Path C:\phon\prep_paper.log -Pattern "PREP PAPER DONE" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 30 }
"$(Get-Date -Format 'HH:mm') prep_paper done" | Add-Content C:\phon\logs\runner.log
# 2. exposure models (same recipe as M0, only the training set differs)
Step "train_M1" "-u train_exposure.py --variant M1"
Step "train_M2" "-u train_exposure.py --variant M2"
# 3. main arms on CUDA for all four models
$TAR = "C:\phon\eval\modelcache\models--tarteel-ai--whisper-base-ar-quran\snapshots\5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"
$M0 = "C:\phon\eval\phonation_study\unseen_voice\checkpoint"
Step "eval_tarteel" "-u run_eval_cuda.py --ckpt $TAR --out C:\phon\results\tarteel"
Step "eval_M0"      "-u run_eval_cuda.py --ckpt $M0 --out C:\phon\results\M0"
Step "eval_M1"      "-u run_eval_cuda.py --ckpt C:\phon\runs\wb_M1\best --out C:\phon\results\M1"
Step "eval_M2"      "-u run_eval_cuda.py --ckpt C:\phon\runs\wb_M2\best --out C:\phon\results\M2"
# 4. round-trip artifact arms
Step "extra_tarteel" "-u run_extra_cuda.py --ckpt $TAR --out C:\phon\results\rt_tarteel"
Step "extra_M0"      "-u run_extra_cuda.py --ckpt $M0 --out C:\phon\results\rt_M0"
Step "extra_M1"      "-u run_extra_cuda.py --ckpt C:\phon\runs\wb_M1\best --out C:\phon\results\rt_M1"
Step "extra_M2"      "-u run_extra_cuda.py --ckpt C:\phon\runs\wb_M2\best --out C:\phon\results\rt_M2"
"$(Get-Date -Format 'HH:mm') RUNNER COMPLETE" | Add-Content C:\phon\logs\runner.log
