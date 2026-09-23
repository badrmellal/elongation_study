Get-Process curl -ErrorAction SilentlyContinue | Stop-Process -Force
$c = Get-Content C:\phon\census_current.json -Raw | ConvertFrom-Json
$rev = $c.revision
$f = $c.files.PSObject.Properties | ForEach-Object { [pscustomobject]@{ Name=$_.Name; Usable=$_.Value.usable_rows; Size=$_.Value.size } }
$val = @($f | Where-Object { $_.Name -like "*validation-*" -and $_.Usable -gt 0 } | Sort-Object Usable -Descending | Select-Object -First 1)
$train = @($f | Where-Object { $_.Name -like "*train-*" -and $_.Usable -gt 0 } | Sort-Object Usable -Descending)
$todo = @(($val + $train) | Where-Object { $p = "C:\phon\everyayah\$($_.Name)"; -not ((Test-Path $p) -and ((Get-Item $p).Length -eq $_.Size)) })
"REMAINING $($todo.Count)"
if ($todo.Count -eq 0) { "ALL COMPLETE"; exit 0 }
$lines = foreach ($x in $todo) { "url = `"https://huggingface.co/datasets/tarteel-ai/everyayah/resolve/$rev/$($x.Name)`""; "output = `"C:/phon/everyayah/$($x.Name)`"" }
Set-Content -Path C:\phon\curl_list.txt -Value $lines -Encoding ascii
curl.exe -sS -L --ssl-no-revoke --create-dirs --retry 8 --retry-all-errors --retry-delay 5 -C - --parallel --parallel-max 3 -K C:\phon\curl_list.txt 2>&1 | Select-Object -Last 3
