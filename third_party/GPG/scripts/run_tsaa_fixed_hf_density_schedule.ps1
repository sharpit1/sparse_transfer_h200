param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^HF[0-9]{12}$')]
    [string]$RunId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$gpgRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = 'C:\Users\raist\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$venv = 'D:\projects\sPGD_transfer\.venv'
$sitePackages = Join-Path $venv 'Lib\site-packages'
$sourceCheckpoint = 'D:\projects\sparse_transfer\atifact\baseline_models\baseline_models\TSAA_netG_-1_res50_train_2.pth'
$sourceSha256 = '1e5c1d4f6f423a22e42ccb1ec17e74ac8cb725f153c3052b94b753e6291b0f54'
$runRoot = Join-Path $gpgRoot (Join-Path 'runs' $RunId)
$outputRoot = Join-Path $runRoot 'training'
$stdoutPath = Join-Path $runRoot 'train.stdout.log'
$stderrPath = Join-Path $runRoot 'train.stderr.log'
$statePath = Join-Path $runRoot 'run_state.json'

function Write-State {
    param([string]$Status, [int]$ExitCode = -1, [string]$Message = '')
    [ordered]@{
        status = $Status
        launcher_pid = $PID
        run_id = $RunId
        epochs = '0-15'
        warmup_epochs = '0-2'
        fixed_lambda_epochs = '3-11'
        adaptive_epochs = '12-15'
        lam_spa_initial = 1.5e-4
        target_density = 0.105
        feature_energy_loss_mode = 'layer1_hf_change'
        feature_energy_loss_lambda = 1.0
        exit_code = $ExitCode
        updated_at = (Get-Date).ToString('o')
        message = $Message
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
}

if (Test-Path -LiteralPath $runRoot) {
    throw "Refusing to reuse existing run directory: $runRoot"
}
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

try {
    foreach ($requiredPath in @($python, $sitePackages, $sourceCheckpoint)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Required path is missing: $requiredPath"
        }
    }
    $observedSha256 = (Get-FileHash -LiteralPath $sourceCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observedSha256 -ne $sourceSha256) {
        throw "Source checkpoint SHA-256 mismatch: $observedSha256"
    }

    $env:VIRTUAL_ENV = $venv
    $env:PYTHONPATH = $sitePackages
    $env:PATH = ((Join-Path $venv 'Scripts'), (Join-Path $sitePackages 'torch\lib'), $env:PATH) -join ';'
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONFAULTHANDLER = '1'
    $env:PYTHONHASHSEED = '42'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:TORCH_HOME = Join-Path $gpgRoot '.cache\torch'
    $env:HF_HOME = Join-Path $gpgRoot '.cache\huggingface'

    $arguments = @(
        '-u', '-B', (Join-Path $gpgRoot 'DDSC_GPG_train.py'),
        '--train_dir', 'C:\Users\raist\data\imagenet\train',
        '--model_type', 'res50',
        '--architecture_mode', 'tsaa',
        '--eps', '10',
        '--target', '-1',
        '--batch_size', '16',
        '--sample_per_class', '0',
        '--n_iters', '1',
        '--epochs', '16',
        '--lr', '1e-4',
        '--lam_1', '1.5e-4',
        '--lam_2', '1e-4',
        '--lam_3', '0',
        '--pb', 'full',
        '--load_CP', 'New',
        '--init_generator_checkpoint', $sourceCheckpoint,
        '--init_generator_checkpoint_sha256', $sourceSha256,
        '--out-dir', $outputRoot,
        '--device', 'cuda:0',
        '--num_workers', '12',
        '--worker_timeout_seconds', '120',
        '--seed', '42',
        '--save_every', '1',
        '--ddsc_mode', 'adaptive',
        '--ddsc_target_density', '0.105',
        '--ddsc_warmup_epochs', '3',
        '--ddsc_control_start_epoch', '12',
        '--ddsc_ema_decay', '0',
        '--ddsc_mass', '1',
        '--ddsc_damping', '0.25',
        '--ddsc_restoring_gain', '1.5e-4',
        '--ddsc_dt', '1',
        '--ddsc_lambda1_min', '0',
        '--intersection_reg_mode', 'off',
        '--intersection_reg_lambda', '0',
        '--layer1_dropout_mode', 'off',
        '--feature_energy_loss_mode', 'layer1_hf_change',
        '--feature_energy_loss_lambda', '1'
    )

    Write-State -Status 'running'
    Push-Location -LiteralPath $gpgRoot
    try {
        & $python @arguments 1>> $stdoutPath 2>> $stderrPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        Write-State -Status 'failed' -ExitCode $exitCode -Message 'Trainer exited unsuccessfully.'
        exit $exitCode
    }
    Write-State -Status 'complete' -ExitCode 0
}
catch {
    Write-State -Status 'failed' -ExitCode 1 -Message $_.Exception.Message
    throw
}
