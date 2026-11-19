# Start the notetaker server automatically when you sign in to Windows, so it comes back by
# itself after a reboot (teammates' recordings wait on their computers until it's up).
#
# Install:    powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
# Remove:     powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove
#
# It registers a task for your user only ("Notetaker server"), run 1 minute after sign-in so
# Docker Desktop and Ollama have started. No administrator rights needed. Also set
# Settings > System > Power > Sleep to "Never" when plugged in, so the PC stays reachable.

param([switch]$Remove)

$name = "Notetaker server"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed the '$name' sign-in task."
    exit 0
}

$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\start.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Minimized -File `"$script`" -NoBrowser" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$trigger.Delay = "PT1M"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -Description "Starts the notetaker (database, API, GPU worker) at sign-in" -Force | Out-Null
Write-Host "Installed: the notetaker starts 1 minute after you sign in. Remove with -Remove."
