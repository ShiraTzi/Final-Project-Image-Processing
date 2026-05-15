param(
    [switch] $RemoveImageArchives
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Cleaning COCO workspace artifacts..."

# Remove annotation zip files (if present)
$zips = @("Image Data\annotations_trainval2017.zip", "Image Data\panoptic_annotations_trainval2017.zip")
foreach ($z in $zips) {
    if (Test-Path $z) {
        Remove-Item $z -Force -Verbose
    }
}

# Remove __MACOSX directories and .DS_Store files
Get-ChildItem -Path "Image Data" -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
    ($_.PSIsContainer -and $_.Name -eq "__MACOSX") -or ($_.PSIsContainer -eq $false -and $_.Name -eq ".DS_Store")
} | ForEach-Object {
    if ($_.PSIsContainer) { Remove-Item $_.FullName -Recurse -Force -Verbose }
    else { Remove-Item $_.FullName -Force -Verbose }
}

if ($RemoveImageArchives) {
    $imgZips = @("Image Data\train2017.zip", "Image Data\val2017.zip", "Image Data\test2017.zip")
    foreach ($f in $imgZips) {
        if (Test-Path $f) { Remove-Item $f -Force -Verbose }
    }
}

Write-Host "Cleanup complete."
