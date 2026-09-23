$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_HOME = "C:\phon\hf"
$py = "C:\phon\venv\Scripts\python.exe"
Start-Process $py -ArgumentList "C:\phon\code\keepawake.py", $PID -WindowStyle Hidden
function Step($name, $argline) {
  "$(Get-Date -Format 'HH:mm') START $name" | Add-Content C:\phon\logs\runner.log
  $p = Start-Process $py -ArgumentList $argline -WorkingDirectory C:\phon\code -NoNewWindow -Wait -PassThru -RedirectStandardOutput "C:\phon\logs\$name.log" -RedirectStandardError "C:\phon\logs\$name.err"
  "$(Get-Date -Format 'HH:mm') END   $name exit $($p.ExitCode)" | Add-Content C:\phon\logs\runner.log
}
$TAR = "C:\phon\eval\modelcache\models--tarteel-ai--whisper-base-ar-quran\snapshots\5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"
$M0 = "C:\phon\eval\phonation_study\unseen_voice\checkpoint"
Step "dose_tarteel" "-u run_rb_dose_cuda.py --ckpt $TAR --out C:\phon\results\dose_tarteel"
Step "dose_M0" "-u run_rb_dose_cuda.py --ckpt $M0 --out C:\phon\results\dose_M0"
Step "dose_M1" "-u run_rb_dose_cuda.py --ckpt C:\phon\runs\wb_M1\best --out C:\phon\results\dose_M1"
Step "dose_M2" "-u run_rb_dose_cuda.py --ckpt C:\phon\runs\wb_M2\best --out C:\phon\results\dose_M2"
"$(Get-Date -Format 'HH:mm') DOSE RUNNER COMPLETE" | Add-Content C:\phon\logs\runner.log
