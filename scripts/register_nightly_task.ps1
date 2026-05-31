#Requires -Version 5.1
<#
.SYNOPSIS
    Registers (or updates) the Windows Scheduled Task that runs the Fortuna
    nightly training orchestrator at 02:00 daily.

.DESCRIPTION
    Creates a SCHTASKS entry with:
      - Daily trigger at 02:00 starting today
      - Wake the computer to run the task
      - Run whether user is logged in or not
      - 6-hour execution limit (orchestrator should finish in ~3 h)
      - Restart on failure (up to 2 attempts, 30-min delay)
      - Highest privileges (needed for SetSuspendState if you want sleep-after)

    Run this script ONCE from an elevated PowerShell:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/register_nightly_task.ps1

    To remove later:
        schtasks /Delete /TN "FortunaNightlyTraining" /F

.PARAMETER TaskName
    Name shown in Task Scheduler. Default: "FortunaNightlyTraining".

.PARAMETER At
    24-h HH:mm time of day to run. Default: "02:00".

.PARAMETER User
    Run as this user. Default: current user (uses 'SYSTEM' if not provided
    interactively, but you generally want your own user for the model files).
#>
param(
    [string]$TaskName = "FortunaNightlyTraining",
    [string]$At = "02:00",
    [string]$User = "$env:USERDOMAIN\$env:USERNAME"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $repo "scripts\nightly_train.ps1"

if (-not (Test-Path $launcher)) {
    throw "Launcher not found: $launcher"
}

Write-Host "Repo:     $repo"
Write-Host "Launcher: $launcher"
Write-Host "Task:     $TaskName"
Write-Host "Schedule: daily at $At"
Write-Host "User:     $User"
Write-Host ""

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`"" `
    -WorkingDirectory $repo

# Daily trigger; start today.
$trigger = New-ScheduledTaskTrigger -Daily -At $At

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

# ``Interactive`` logon does NOT require admin elevation. The task runs only
# when the user is logged in (which is the case for a PC put to sleep), and
# WakeToRun still works in that state. If you want it to run regardless of
# logon state, re-run this script from an elevated PowerShell — it will
# automatically upgrade to LogonType=S4U.
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
    $principal = New-ScheduledTaskPrincipal `
        -UserId $User `
        -LogonType S4U `
        -RunLevel Highest
    Write-Host "Detected admin -> using S4U logon (runs even when not logged in)."
} else {
    $principal = New-ScheduledTaskPrincipal `
        -UserId $User `
        -LogonType Interactive `
        -RunLevel Limited
    Write-Host "Non-admin -> using Interactive logon (PC must be at-least logged-in/sleeping)."
}

$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Fortuna RL/ML nightly training orchestrator"

# Re-register: remove first if present.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'."
}

Register-ScheduledTask -TaskName $TaskName -InputObject $task -User $User | Out-Null
Write-Host "Registered '$TaskName' (daily at $At, wake-to-run)."

# Show summary
$reg = Get-ScheduledTask -TaskName $TaskName
$info = $reg | Get-ScheduledTaskInfo
Write-Host ""
Write-Host "Next run time: $($info.NextRunTime)"
Write-Host "State:         $($reg.State)"
Write-Host ""
Write-Host "Manual trigger: schtasks /Run /TN `"$TaskName`""
Write-Host "Disable:        schtasks /Change /TN `"$TaskName`" /Disable"
Write-Host "Delete:         schtasks /Delete /TN `"$TaskName`" /F"
