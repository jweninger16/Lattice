$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Lattice.lnk"
$TargetPath = Join-Path $ProjectRoot "lattice\lattice_run.bat"
$IconPath = Join-Path $ProjectRoot "lattice\lattice.ico"
$WorkingDir = $ProjectRoot

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.WorkingDirectory = $WorkingDir
$Shortcut.IconLocation = "$IconPath, 0"
$Shortcut.Description = "Lattice - Automated Trading Engine"
$Shortcut.WindowStyle = 1
$Shortcut.Save()

Write-Host "  Lattice shortcut created on your desktop!"
Write-Host "  Double-click Lattice to launch."
