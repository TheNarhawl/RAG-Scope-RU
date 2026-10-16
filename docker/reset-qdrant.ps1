# Сносит контейнер и volume, поднимает Qdrant заново, ждёт готовности.
# Запуск из корня проекта: .\docker\restart-qdrant.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Останавливаем и удаляем контейнер qdrant..." -ForegroundColor Cyan
docker rm -f qdrant 2>$null | Out-Null

Write-Host "==> Удаляем volume qdrant_storage..." -ForegroundColor Cyan
docker volume rm qdrant_storage 2>$null | Out-Null

Write-Host "==> Запускаем Qdrant заново..." -ForegroundColor Cyan
docker run -d --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage `
  qdrant/qdrant:latest | Out-Null

Write-Host "==> Ждём готовности Qdrant..." -ForegroundColor Cyan
$ready = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://localhost:6333" -TimeoutSec 2
        if ($r.title -match "qdrant") {
            $ready = $true
            break
        }
    } catch {
        # ещё не готов
    }
    Write-Host "    ... $i сек" -ForegroundColor DarkGray
}

if ($ready) {
    Write-Host "==> Qdrant готов" -ForegroundColor Green
    Invoke-RestMethod -Uri "http://localhost:6333" | Format-List
} else {
    Write-Host "==> Qdrant не поднялся за 30 секунд" -ForegroundColor Red
    Write-Host "    Логи:" -ForegroundColor Yellow
    docker logs qdrant --tail 30
    exit 1
}