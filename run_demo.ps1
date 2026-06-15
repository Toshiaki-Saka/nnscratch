#Requires -Version 7.0
<#
.SYNOPSIS
    nnscratch をビルドし、学習シミュレーションとアニメーションを実行する。
.DESCRIPTION
    1. CMake でビルド
    2. ctest でテスト実行
    3. from_scratch デモ: エポックごとにリアルタイムプログレスバーをアニメーション表示
    4. 学習曲線を ASCII チャートで可視化
    5. compare デモ: オプティマイザ / 活性化関数 / アーキテクチャ比較
    6. 比較結果をバーチャートで表示
.PARAMETER BuildType
    CMake ビルドタイプ (Release / Debug)。デフォルト: Release
.PARAMETER BuildDir
    ビルドディレクトリ名。デフォルト: build
.PARAMETER OutDir
    デモ出力（CSV, PGM）ディレクトリ名。デフォルト: output
.PARAMETER SkipBuild
    ビルドをスキップしてデモのみ実行する。
.PARAMETER SkipTests
    ctest をスキップする。
.EXAMPLE
    .\run_demo.ps1
.EXAMPLE
    .\run_demo.ps1 -SkipBuild
#>
[CmdletBinding()]
param(
    [string]$BuildType = "Release",
    [string]$BuildDir  = "build",
    [string]$OutDir    = "output",
    [switch]$SkipBuild,
    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── ANSI / コンソール初期化 ───────────────────────────────────────────────────
# Windows 11 の VT100 サポートを有効化
if ($IsWindows) {
    $null = [System.Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Add-Type -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool SetConsoleMode(IntPtr hConsoleHandle, int mode);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out int lpMode);
[DllImport("kernel32.dll", SetLastError = true)]
public static extern IntPtr GetStdHandle(int nStdHandle);
"@ -Namespace Win32 -Name Console -ErrorAction SilentlyContinue

    try {
        $handle = [Win32.Console]::GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        $mode   = 0
        [void][Win32.Console]::GetConsoleMode($handle, [ref]$mode)
        [void][Win32.Console]::SetConsoleMode($handle, $mode -bor 4)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    } catch {}
}

$ESC = [char]27
function C($code) { "${ESC}[${code}m" }

$Bold    = C 1
$Dim     = C 2
$Reset   = C 0
$Green   = C "32"
$Yellow  = C "33"
$Cyan    = C "36"
$Magenta = C "35"
$Red     = C "31"
$Blue    = C "34"
$White   = C "97"

# カーソル移動
function Up($n)    { "${ESC}[${n}A" }
function ClearLine { "${ESC}[2K`r" }

# ── ユーティリティ ────────────────────────────────────────────────────────────
function Write-Color {
    param([string]$Text, [string]$Color = $White, [switch]$NoNewLine)
    if ($NoNewLine) { Write-Host -NoNewline "${Color}${Text}${Reset}" }
    else            { Write-Host             "${Color}${Text}${Reset}" }
}

function Write-Header {
    param([string]$Title, [string]$Icon = "◆")
    $width = 68
    $bar   = "─" * $width
    Write-Host ""
    Write-Color "  ┌${bar}┐" $Cyan
    $pad = " " * [Math]::Max(0, $width - $Title.Length - 3)
    Write-Color "  │  ${Icon} ${Title}${pad}│" $Cyan
    Write-Color "  └${bar}┘" $Cyan
    Write-Host ""
}

function Make-Bar {
    param([double]$Ratio, [int]$Width = 20)
    $filled = [int]([Math]::Min(1.0, [Math]::Max(0.0, $Ratio)) * $Width)
    ("█" * $filled) + ("░" * ($Width - $filled))
}

function Acc-Color {
    param([double]$Acc)
    if ($Acc -ge 0.95) { return $Green }
    if ($Acc -ge 0.80) { return $Yellow }
    return $Cyan
}

# ── スピナー (バックグラウンドジョブ) ────────────────────────────────────────
$script:SpinnerActive = $false
$script:SpinnerJob    = $null

function Start-Spinner {
    param([string]$Msg = "Processing")
    $script:SpinnerActive = $true
    $script:SpinnerJob = Start-Job -ScriptBlock {
        param($msg)
        $frames = @("⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏")
        $i = 0
        $ESC = [char]27
        while ($true) {
            $f = $frames[$i % $frames.Count]
            [System.Console]::Write("`r${ESC}[36m  ${f} ${msg}...${ESC}[0m")
            Start-Sleep -Milliseconds 80
            $i++
        }
    } -ArgumentList $Msg
}

function Stop-Spinner {
    if ($script:SpinnerJob) {
        Stop-Job  $script:SpinnerJob -ErrorAction SilentlyContinue
        Remove-Job $script:SpinnerJob -ErrorAction SilentlyContinue
        $script:SpinnerJob = $null
    }
    [System.Console]::Write("`r$(' ' * 60)`r")
}

# ── CMake / ビルド ────────────────────────────────────────────────────────────
function Assert-Cmake {
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmake) {
        Write-Color "  ERROR: cmake が見つかりません。CMake をインストールして PATH に追加してください。" $Red
        exit 1
    }
    return $cmake.Source
}

function Build-Project {
    param([string]$SrcDir, [string]$BldDir, [string]$Type)

    Write-Header "ビルド (CMake)" "🔨"

    $cmake = Assert-Cmake
    Write-Color "  cmake  : $cmake" $Dim
    Write-Color "  Source : $SrcDir" $Dim
    Write-Color "  Build  : $BldDir  [$Type]" $Dim
    Write-Host ""

    Write-Color "  Configure..." $Yellow
    & $cmake -S $SrcDir -B $BldDir -DCMAKE_BUILD_TYPE=$Type
    if ($LASTEXITCODE -ne 0) { Write-Color "  ERROR: cmake configure 失敗" $Red; exit 1 }

    Write-Host ""
    Write-Color "  Build..." $Yellow
    & $cmake --build $BldDir --config $Type
    if ($LASTEXITCODE -ne 0) { Write-Color "  ERROR: cmake build 失敗" $Red; exit 1 }

    Write-Host ""
    Write-Color "  ✓ ビルド成功" $Green
}

# ── 実行ファイル検索 ──────────────────────────────────────────────────────────
function Find-Exe {
    param([string]$BldDir, [string]$Name, [string]$Type)
    foreach ($sub in @($Type, "Release", "Debug", "")) {
        foreach ($ext in @(".exe", "")) {
            $candidate = if ($sub) {
                Join-Path $BldDir "${sub}\${Name}${ext}"
            } else {
                Join-Path $BldDir "${Name}${ext}"
            }
            if (Test-Path $candidate) { return $candidate }
        }
    }
    Write-Color "  ERROR: $Name 実行ファイルが $BldDir に見つかりません" $Red
    exit 1
}

# ── テスト ─────────────────────────────────────────────────────────────────────
function Run-Tests {
    param([string]$BldDir, [string]$Type)
    Write-Header "テスト (ctest)" "🧪"
    & ctest --test-dir $BldDir -C $Type --output-on-failure
    if ($LASTEXITCODE -ne 0) {
        Write-Color "  ✗ テストに失敗があります" $Red
    } else {
        Write-Color "  ✓ 全テスト通過" $Green
    }
}

# ── from_scratch アニメーション ───────────────────────────────────────────────
function Run-FromScratch {
    param([string]$ExePath, [string]$CsvPath, [string]$OutputDir)

    Write-Header "Part 1 — from_scratch : ランダム → 学習済み MLP" "🧠"

    $totalEpochs  = 60
    $drewBars     = $false   # 前フレームが描画済みか
    $barLines     = 5        # 描画する行数

    # エポック行の正規表現
    $epochRx = [regex]'epoch\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*train_acc\s+([\d.]+)%\s*\|\s*test_acc\s+([\d.]+)%'

    & $ExePath $CsvPath $OutputDir 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $m    = $epochRx.Match($line)

        if ($m.Success) {
            $ep       = [int]   $m.Groups[1].Value
            $loss     = [double]$m.Groups[2].Value
            $trainAcc = [double]$m.Groups[3].Value / 100.0
            $testAcc  = [double]$m.Groups[4].Value / 100.0

            $pct      = $ep / $totalEpochs
            $pBar     = Make-Bar $pct       30
            $trainBar = Make-Bar $trainAcc  22
            $testBar  = Make-Bar $testAcc   22

            $tc = Acc-Color $trainAcc
            $sc = Acc-Color $testAcc
            $pc = if ($pct -ge 0.5) { $Green } else { $Cyan }

            # 前のバーを上書き
            if ($drewBars) {
                Write-Host -NoNewline (Up $barLines)
            }

            # ─── プログレス表示（$barLines 行） ───
            Write-Host "$(ClearLine)  ${Bold}Epoch $("{0,3}" -f $ep) / ${totalEpochs}${Reset}  [${pc}${pBar}${Reset}]  Loss: ${Cyan}$($loss.ToString('F4'))${Reset}"
            Write-Host "$(ClearLine)  Train  [${tc}${trainBar}${Reset}] ${tc}$($m.Groups[3].Value)%${Reset}"
            Write-Host "$(ClearLine)  Test   [${sc}${testBar}${Reset}] ${sc}$($m.Groups[4].Value)%${Reset}"
            Write-Host "$(ClearLine)  $($Dim)──────────────────────────────────────────────────────────$($Reset)"
            Write-Host "$(ClearLine)"

            $drewBars = $true
        }
        elseif ($line -match 'Untrained test accuracy:\s*([\d.]+)') {
            Write-Host "  ${Yellow}初期精度 (未学習): $($Matches[1])%${Reset}  ← ランダム予測とほぼ同じ"
            Write-Host ""
        }
        elseif ($line -match 'Final test accuracy:\s*([\d.]+)') {
            Write-Host ""
            Write-Color "  ✓ 最終テスト精度: $($Matches[1])%" $Green
        }
        elseif ($line -match 'Wrote:') {
            Write-Color "  $line" $Dim
        }
        elseif ($line -match 'Loading|Training') {
            Write-Color "  $line" $Yellow
        }
        elseif ($line -match '^={3,}|^[-]{3,}') {
            # セクション区切り — スキップ
        }
        elseif ($line.Trim() -ne '') {
            Write-Host "  $line"
        }
    }
}

# ── 学習曲線 ASCII チャート ───────────────────────────────────────────────────
function Show-LearningCurve {
    param([string]$CsvPath)

    if (-not (Test-Path $CsvPath)) { return }
    $rows = Import-Csv $CsvPath
    if ($rows.Count -lt 2) { return }

    Write-Header "学習曲線 (Test Accuracy)" "📈"

    $testAccs = $rows | ForEach-Object { [double]$_.test_acc }
    $losses   = $rows | ForEach-Object { [double]$_.train_loss }
    $epochs   = $rows | ForEach-Object { [int]$_.epoch }

    $chartH = 12
    $chartW = [Math]::Min($rows.Count, 58)
    $step   = [Math]::Max(1, [int][Math]::Ceiling($rows.Count / $chartW))
    $idxs   = for ($i = 0; $i -lt $rows.Count; $i += $step) { $i }

    # テスト精度チャート
    Write-Host "  ${Dim}Test Accuracy${Reset}"
    for ($row = $chartH; $row -ge 0; $row--) {
        $thresh = $row / $chartH
        $yLabel = if ($row % 4 -eq 0) { "${Dim}{0,5:P0}${Reset} │" -f $thresh }
                  else                 { "      │" }
        $line = "  $yLabel"
        foreach ($idx in $idxs) {
            $v = $testAccs[$idx]
            if ($v -ge $thresh) {
                $col = if ($v -ge 0.95) { $Green } elseif ($v -ge 0.80) { $Yellow } else { $Cyan }
                $line += "${col}█${Reset}"
            } else {
                $line += " "
            }
        }
        Write-Host $line
    }
    $xBar = "      └" + ("─" * $idxs.Count)
    Write-Host "  $xBar"
    $midPad = " " * ([int]($idxs.Count / 2) - 5)
    Write-Host "       0${midPad}epoch${midPad}$($epochs[-1])"
    Write-Host ""

    # ロスチャート
    $maxLoss = ($losses | Measure-Object -Maximum).Maximum
    Write-Host "  ${Dim}Training Loss${Reset}"
    for ($row = $chartH; $row -ge 0; $row--) {
        $thresh = ($row / $chartH) * $maxLoss
        $yLabel = if ($row % 4 -eq 0) { "${Dim}{0,5:F2}${Reset} │" -f $thresh }
                  else                 { "      │" }
        $line = "  $yLabel"
        foreach ($idx in $idxs) {
            $v = $losses[$idx]
            $pct = if ($maxLoss -gt 0) { $v / $maxLoss } else { 0 }
            if ($pct -ge ($row / $chartH)) {
                $col = if ($pct -lt 0.3) { $Green } elseif ($pct -lt 0.6) { $Yellow } else { $Magenta }
                $line += "${col}█${Reset}"
            } else {
                $line += " "
            }
        }
        Write-Host $line
    }
    $xBar2 = "      └" + ("─" * $idxs.Count)
    Write-Host "  $xBar2"
    Write-Host "       0${midPad}epoch${midPad}$($epochs[-1])"
    Write-Host ""
}

# ── compare デモ ──────────────────────────────────────────────────────────────
function Run-Compare {
    param([string]$ExePath, [string]$CsvPath, [string]$OutputDir)

    Write-Header "Part 2 — compare : オプティマイザ / 活性化関数 / アーキテクチャ" "⚡"

    # compare は verbose=false なので、実験終了後に結果テーブルだけ出力される。
    # スピナー表示用にバックグラウンドで exe を実行し、出力をキャプチャする。

    $tmpOut = Join-Path $OutputDir "_compare_out.txt"
    $proc = Start-Process -FilePath $ExePath `
                          -ArgumentList @($CsvPath, $OutputDir) `
                          -RedirectStandardOutput $tmpOut `
                          -RedirectStandardError  "$OutputDir\_compare_err.txt" `
                          -NoNewWindow `
                          -PassThru

    # スピナー (各実験がどれくらいかを示す進捗)
    $spinFrames  = @("⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏")
    $expNames    = @("Experiment 1: Optimizers (SGD / Momentum / Adam) [40 epochs × 3]",
                     "Experiment 2: Activations (ReLU / Tanh / Sigmoid) [40 epochs × 3]",
                     "Experiment 3: Architecture (shallow / MLP / CNN) [25 epochs × 3]")
    $expIdx = 0
    $frame  = 0

    Write-Host ""
    while (-not $proc.HasExited) {
        $f = $spinFrames[$frame % $spinFrames.Count]

        # 出力ファイルの行数で実験進捗を推定
        $linesNow = 0
        if (Test-Path $tmpOut) {
            $linesNow = (Get-Content $tmpOut -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
        }
        # 各実験後に print_table が数行出力される
        $expIdx = [Math]::Min(2, [int]($linesNow / 8))

        $expMsg = $expNames[[Math]::Min($expIdx, 2)]
        Write-Host -NoNewline "${ESC}[2K`r  ${Cyan}${f}${Reset} ${expMsg}"

        Start-Sleep -Milliseconds 80
        $frame++
    }
    Write-Host -NoNewline "`r$(' ' * 80)`r"  # スピナー消去

    # 出力をパース＆カラー表示
    $tableRx  = [regex]'^\s*(\S+)\s+([\d.]+)%\s+([\d.]+)%\s+(.+)$'
    $headerRx = [regex]'name\s+final'
    $expRx    = [regex]'=== Experiment (\d+):'

    if (Test-Path $tmpOut) {
        $outLines = Get-Content $tmpOut
        foreach ($line in $outLines) {
            if ($expRx.IsMatch($line)) {
                $n = $expRx.Match($line).Groups[1].Value
                Write-Host ""
                Write-Color "  ◆ $($line.Trim())" $Yellow
            }
            elseif ($headerRx.IsMatch($line)) {
                Write-Host ""
                Write-Color "    $($line.Trim())" $Bold
            }
            elseif ($line -match '^-{10,}') {
                Write-Color "    $($line.Trim())" $Dim
            }
            elseif ($tableRx.IsMatch($line)) {
                $tm = $tableRx.Match($line)
                $acc = [double]$tm.Groups[2].Value
                $col = if ($acc -ge 95) { $Green } elseif ($acc -ge 85) { $Yellow } else { $Cyan }
                Write-Host "    ${col}$($line.Trim())${Reset}"
            }
            elseif ($line -match 'Wrote') {
                Write-Color "  $($line.Trim())" $Dim
            }
            elseif ($line.Trim() -ne '') {
                Write-Host "  $($line.Trim())"
            }
        }
    }

    # 一時エラーファイル除去
    Remove-Item "$OutputDir\_compare_err.txt" -ErrorAction SilentlyContinue
    Remove-Item $tmpOut -ErrorAction SilentlyContinue
}

# ── 比較バーチャート ──────────────────────────────────────────────────────────
function Show-CompareChart {
    param([string]$CsvPath, [string]$Title)

    if (-not (Test-Path $CsvPath)) { return }
    $rows = Import-Csv $CsvPath
    if ($rows.Count -eq 0) { return }

    $names = $rows | Select-Object -ExpandProperty name -Unique
    $finals = @{}
    foreach ($n in $names) {
        $last = ($rows | Where-Object { $_.name -eq $n } | Select-Object -Last 1)
        $finals[$n] = [double]$last.test_acc
    }

    Write-Color "  ${Bold}${Title}${Reset}" $Cyan
    $barW = 36
    foreach ($n in ($finals.Keys | Sort-Object)) {
        $acc = $finals[$n]
        $bar = Make-Bar $acc $barW
        $col = Acc-Color $acc
        $label = $n.PadRight(18)
        Write-Host ("  {0} [{1}{2}{3}] {4}" -f $label, $col, $bar, $Reset, ("${col}$($acc.ToString('P1'))${Reset}"))
    }
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BldPath   = Join-Path $ScriptDir $BuildDir
$OutPath   = Join-Path $ScriptDir $OutDir
$DataCsv   = Join-Path $ScriptDir "data\digits.csv"

# ── バナー ─────────────────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Color "  ╔══════════════════════════════════════════════════════════════════╗" $Cyan
Write-Color "  ║                                                                  ║" $Cyan
Write-Color "  ║   nnscratch  —  C++20 ニューラルネット デモランナー             ║" $Cyan
Write-Color "  ║                                                                  ║" $Cyan
Write-Color "  ╚══════════════════════════════════════════════════════════════════╝" $Cyan
Write-Host ""
Write-Color "  Source  : $ScriptDir"  $Dim
Write-Color "  Build   : $BldPath  [$BuildType]" $Dim
Write-Color "  Output  : $OutPath" $Dim
Write-Host ""

# 出力ディレクトリ作成
if (-not (Test-Path $OutPath)) { New-Item -ItemType Directory -Path $OutPath | Out-Null }

# 1. ビルド
if (-not $SkipBuild) {
    Build-Project -SrcDir $ScriptDir -BldDir $BldPath -Type $BuildType
}

# 2. テスト
if (-not $SkipTests) {
    Run-Tests -BldDir $BldPath -Type $BuildType
}

# 3. 実行ファイル探索
$fsExe  = Find-Exe $BldPath "from_scratch" $BuildType
$cmpExe = Find-Exe $BldPath "compare"      $BuildType
Write-Color "  from_scratch : $fsExe" $Dim
Write-Color "  compare      : $cmpExe" $Dim
Write-Host ""
Read-Host "  Enter キーを押してデモを開始します"

# 4. from_scratch デモ (リアルタイムアニメーション)
Run-FromScratch -ExePath $fsExe -CsvPath $DataCsv -OutputDir $OutPath

# 5. 学習曲線 ASCII チャート
Show-LearningCurve -CsvPath (Join-Path $OutPath "learning_curve.csv")

Read-Host "  Enter キーで比較実験を開始します"

# 6. compare デモ (スピナー → 結果カラー表示)
Run-Compare -ExePath $cmpExe -CsvPath $DataCsv -OutputDir $OutPath

# 7. 比較サマリーチャート
Write-Header "比較サマリー" "📊"
Show-CompareChart (Join-Path $OutPath "cmp_optimizers.csv")   "Optimizers"
Show-CompareChart (Join-Path $OutPath "cmp_activations.csv")  "Activations"
Show-CompareChart (Join-Path $OutPath "cmp_architecture.csv") "Architectures"

Write-Host ""
Write-Color "  ✓ 完了。出力ファイル: $OutPath" $Green
Write-Color "    - learning_curve.csv   (from_scratch 学習曲線)" $Dim
Write-Color "    - learned_features.pgm (第1層の重みビジュアル)" $Dim
Write-Color "    - cmp_optimizers.csv   (オプティマイザ比較)" $Dim
Write-Color "    - cmp_activations.csv  (活性化関数比較)" $Dim
Write-Color "    - cmp_architecture.csv (アーキテクチャ比較)" $Dim
Write-Color "    - cnn_filters.pgm      (CNN フィルタビジュアル)" $Dim
Write-Host ""
