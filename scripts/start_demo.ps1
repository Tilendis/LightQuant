# LightQuant 演示环境启动脚本
# 作用：确保 8014 上跑的是当前代码，健康检查通过后打开浏览器。
# 用法：双击仓库根目录的「启动演示.bat」

param(
    # 只起服务、不自动开浏览器（自检或远程调试时用）
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $root '.venv\Scripts\python.exe'
$port = 8014
$base = "http://127.0.0.1:$port"

function Test-Health {
    try {
        $r = Invoke-WebRequest -Uri "$base/api/health" -TimeoutSec 3 -UseBasicParsing
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $py)) {
    Write-Host "找不到解释器：$py"
    Write-Host "请确认 .venv 已建好。"
    exit 1
}

# 本地密钥（可选）：每行 KEY=VALUE。文件放在用户目录而不是仓库里，
# 这样 key 既不进 git、也不写进任何源码或文档。没有这个文件就跳过，
# 服务照常运行，只是发帖审核退回规则层（接口会如实说明）。
# 示例内容：
#   MODERATOR_PROVIDER=dashscope
#   DASHSCOPE_API_KEY=<你自己的 key>
$secretFile = Join-Path $env:USERPROFILE '.lightquant\secrets.env'
if (Test-Path -LiteralPath $secretFile) {
    $loaded = 0
    Get-Content -LiteralPath $secretFile -Encoding utf8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $k, $v = $line.Split('=', 2)
            if ($k -and $v) {
                Set-Item -Path ("Env:" + $k.Trim()) -Value $v.Trim()
                $loaded++
            }
        }
    }
    Write-Host "已加载本地密钥配置（$loaded 项）。"
}

if (Test-Health) {
    Write-Host "服务已在 $port 运行，直接打开页面。"
} else {
    # 端口上若有旧进程（改了代码却没重启），先结束掉，否则新进程 bind 失败会静默退出
    $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match "--port $port" }
    if ($busy) {
        $ids = $busy | Select-Object -ExpandProperty ProcessId
        Write-Host ("端口被旧进程占用，先结束：" + ($ids -join ', '))
        Stop-Process -Id $ids -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }

    $log = Join-Path $env:TEMP 'lightquant_uvicorn.log'
    $err = Join-Path $env:TEMP 'lightquant_uvicorn.err.log'
    $argList = @('-m', 'uvicorn', 'app.api:app', '--host', '127.0.0.1', '--port', "$port")
    Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList $argList -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $err

    Write-Host "正在启动，等待健康检查…"
    $ok = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 800
        if (Test-Health) { $ok = $true; break }
    }
    if (-not $ok) {
        Write-Host "启动失败。错误日志：$err"
        if (Test-Path -LiteralPath $err) { Get-Content -LiteralPath $err -Tail 25 }
        exit 1
    }
    Write-Host "服务就绪：$base"
}

# 回测引擎预热是后台线程，torch 冷启动约 30 秒。等它就绪再开页面，
# 否则第一屏顶栏会写"引擎预热中"，演示时第一次提交回测还要干等。
for ($i = 0; $i -lt 60; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "$base/api/health" -TimeoutSec 3
        if ($h.engine_ready) {
            Write-Host "回测引擎已预热。"
            break
        }
    } catch { }
    Start-Sleep -Seconds 1
}

if (-not $NoBrowser) {
    Start-Process "$base/"
}
