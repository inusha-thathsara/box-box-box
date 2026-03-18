# Test neural network model accuracy on 100 test cases
# Usage: PowerShell.exe -ExecutionPolicy Bypass -File test_nn_model.ps1

$ErrorActionPreference = 'Stop'
Set-Location 'e:\Documents\Projects\sansahackothon\box-box-box'

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║        NEURAL NETWORK MODEL TESTING                           ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

# Function to test model with env var
function Test-Model([string]$Name, [string]$EnvValue) {
    Write-Host "Testing: $Name" -ForegroundColor Cyan
    
    if ($EnvValue) {
        $env:BOXBOXBOX_USE_NN = $EnvValue
    } else {
        if (Test-Path Env:BOXBOXBOX_USE_NN) {
            Remove-Item Env:BOXBOXBOX_USE_NN
        }
    }
    
    $passed = 0
    $failed = 0
    $errors = @()
    
    $inputs = Get-ChildItem data/test_cases/inputs/test_*.json | Sort-Object Name
    $count = 0
    
    foreach ($inputFile in $inputs) {
        $count++
        if ($count -le 5 -or $count -gt 95) {
            Write-Host -NoNewline "."
        }
        
        $expectedPath = Join-Path 'data/test_cases/expected_outputs' ($inputFile.BaseName + '.json')
        $expected = Get-Content $expectedPath -Raw | ConvertFrom-Json
        
        try {
            $outText = Get-Content $inputFile.FullName -Raw | python solution/race_simulator.py
            $output = $outText | ConvertFrom-Json
            
            $match = $true
            for ($i = 0; $i -lt 20; $i++) {
                if ($output.finishing_positions[$i] -ne $expected.finishing_positions[$i]) {
                    $match = $false
                    break
                }
            }
            
            if ($match) {
                $passed++
            } else {
                $failed++
            }
        } catch {
            $failed++
            $errors += "Error on $($inputFile.Name): $_"
        }
    }
    
    Write-Host ""
    $accuracy = [math]::Round(($passed * 100.0 / ($passed + $failed)), 2)
    Write-Host "  Result: PASSED=$passed FAILED=$failed ACCURACY=$accuracy%" -ForegroundColor Yellow
    Write-Host ""
    
    return @{
        Name = $Name
        Passed = $passed
        Failed = $failed
        Accuracy = $accuracy
    }
}

# Test baseline
Write-Host "▶ BASELINE (pure physics, no models)" -ForegroundColor Green
$baseline = Test-Model "Baseline (no NN/Linear)" ""

# Test NN
Write-Host "▶ NEURAL NETWORK" -ForegroundColor Green
if (Test-Path 'solution/nn_model.pth') {
    $nn = Test-Model "NN Model" "1"
} else {
    Write-Host "NN model not found at solution/nn_model.pth" -ForegroundColor Red
    Write-Host "Run: python solution/train_nn_model.py --epochs 10 --output solution/nn_model.pth" -ForegroundColor Yellow
    $nn = @{}
}

# Summary
Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║                         SUMMARY                               ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

Write-Host "Baseline:          $($baseline.Accuracy)% ($($baseline.Passed)/$($baseline.Passed + $baseline.Failed))" -ForegroundColor Cyan
if ($nn.Accuracy) {
    $gain = $nn.Accuracy - $baseline.Accuracy
    $sign = if ($gain -gt 0) { "+" } else { "" }
    $color = if ($gain -gt 0) { "Green" } else { "Red" }
    Write-Host "NN Model:          $($nn.Accuracy)% ($($nn.Passed)/$($nn.Passed + $nn.Failed))" -ForegroundColor $color
    Write-Host "Improvement:       $sign$gain%" -ForegroundColor $color
} else {
    Write-Host "NN Model:          Not tested" -ForegroundColor Gray
}

Write-Host ""
if (Test-Path Env:BOXBOXBOX_USE_NN) {
    Remove-Item Env:BOXBOXBOX_USE_NN
}
