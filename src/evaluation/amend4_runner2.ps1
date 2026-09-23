$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_HOME = "C:\phon\hf"
$py = "C:\phon\venv\Scripts\python.exe"
$LOG = "C:\phon\logs\runner.log"
# Keep the machine awake from this PowerShell thread. No python helper: the UGC runner counts any
# python.exe under C:\phon as a busy owner job, and an idle helper would stall it.
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][W.P]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function Say($m) { "$(Get-Date -Format 'HH:mm') $m" | Add-Content $LOG }
function UgcProcs {
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -like '*ugc-creation*' })
}
function UgcActive { (UgcProcs).Count -gt 0 }
function GpuMiB { [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) }
# UGC work has priority: start only after 2 consecutive minutes with no UGC process and an idle card.
function WaitClear {
  $ok = 0; $said = $false
  while ($ok -lt 4) {
    $procs = UgcProcs; $u = $procs.Count -gt 0; $m = GpuMiB
    $what = ($procs | ForEach-Object { "$($_.ProcessId):" + $_.CommandLine.Substring(0, [Math]::Min(90, $_.CommandLine.Length)) }) -join ' | '
    "$(Get-Date -Format 'HH:mm:ss') ok=$ok ugc=$u used=$m $what" | Add-Content C:\phon\logs\gate.log
    if ($u -or $m -ge 1500) {
      if (-not $said) { Say "WAIT gpu not free (ugc=$u used=$m MiB)"; $said = $true }
      $ok = 0
    } else { $ok++ }
    Start-Sleep 30
  }
  if ($said) { Say "GPU free for 2 min, proceeding" }
}
function TrimPartial($f) {
  if (Test-Path $f) {
    $b = [IO.File]::ReadAllBytes($f)
    if ($b.Length -and $b[-1] -ne 10) {
      $i = [Array]::LastIndexOf($b, [byte]10); $fs = [IO.File]::Open($f, 'Open'); $fs.SetLength($i + 1); $fs.Close()
      Say "trimmed partial last line of $f"
    }
  }
}
# Run one step; if a UGC job appears mid-step, kill this step and redo it after UGC finishes.
# Evaluation steps resume per condition; a preempted or failed training run is set aside and restarted.
# A step already logged as "END <name> exit 0" is skipped, so this runner can be restarted at any time.
# A step that fails is retried once, then logged FAILED and the queue moves on.
function Step($name, $argline, $obs, $rundir) {
  if (Select-String -Path $LOG -Pattern "END   $name exit 0" -SimpleMatch -Quiet) { Say "SKIP  $name (already done)"; return }
  $try = 0; $fails = 0
  while ($true) {
    WaitClear
    if ($obs) { TrimPartial $obs }
    $try++
    Say "START $name (attempt $try)"
    $p = Start-Process $py -ArgumentList $argline -WorkingDirectory C:\phon\code -NoNewWindow -PassThru `
         -RedirectStandardOutput "C:\phon\logs\$name`_$try.log" -RedirectStandardError "C:\phon\logs\$name`_$try.err"
    $null = $p.Handle
    $pre = $false
    while (-not $p.HasExited) {
      Start-Sleep 20
      if (UgcActive) { taskkill /PID $p.Id /T /F | Out-Null; $pre = $true; break }
    }
    if (-not $pre) {
      $p.WaitForExit(); $code = $p.ExitCode; Say "END   $name exit $code"
      if ($code -eq 0) { return }
      $fails++
      if ($fails -ge 2) { Say "FAILED $name after 2 attempts, moving on"; return }
      Say "RETRY $name"
      if ($rundir -and (Test-Path $rundir)) { Rename-Item $rundir "$(Split-Path $rundir -Leaf)_failed_$(Get-Date -Format 'MMddHHmm')" }
      continue
    }
    Say "PREEMPTED $name (UGC job appeared); will redo after it finishes"
    Start-Sleep 5
    if ($rundir -and (Test-Path $rundir)) { Rename-Item $rundir "$(Split-Path $rundir -Leaf)_preempted_$(Get-Date -Format 'MMddHHmm')" }
  }
}

$TAR = "C:\phon\eval\modelcache\models--tarteel-ai--whisper-base-ar-quran\snapshots\5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c"
$M0 = "C:\phon\eval\phonation_study\unseen_voice\checkpoint"
# real-file copy of openai/whisper-large-v3 (HF-cache symlinks break through the cache junction); config dtype set to float32
$V3 = "C:\phon\models\whisper-large-v3"
$CONF = "Alafasy,Sudais,Shuraym,Dussary,Rifai"
$R = "C:\phon\results"
Say "AMEND4 RUNNER 2 (UGC-yielding, restart-safe) started"
# F2 fresh reciters
foreach ($pair in @(@("tarteel",$TAR), @("M0",$M0), @("M1","C:\phon\runs\wb_M1\best"), @("M2","C:\phon\runs\wb_M2\best"))) {
  $n = $pair[0]
  Step "fresh_$n" "-u run_eval_cuda2.py --manifest manifest_fresh.json --ckpt $($pair[1]) --out $R\fresh_$n --resume" "$R\fresh_$n\observations.jsonl" $null
}
# F5 English
Step "libri" "-u libri_eval.py" $null $null
# F4 Whisper-large-v3 (confirmatory reciters)
Step "large_v3" "-u run_eval_cuda2.py --ckpt $V3 --out $R\large_v3 --reciters $CONF --resume" "$R\large_v3\observations.jsonl" $null
# F1 seeds
foreach ($s in 2, 3) { foreach ($v in "M1", "M2") {
  $run = "wb_$($v)_s$s"
  Step "train_$($v)_s$s" "-u train_exposure2.py --variant $v --order-seed $s --run $run" $null "C:\phon\runs\$run"
  Step "eval_$($v)_s$s" "-u run_eval_cuda2.py --ckpt C:\phon\runs\$run\best --out $R\$($v)_s$s --reciters $CONF --resume" "$R\$($v)_s$s\observations.jsonl" $null
} }
Say "AMEND4 RUNNER COMPLETE"
