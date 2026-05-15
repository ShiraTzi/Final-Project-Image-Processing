<#
Project tasks (added by request):
1. Object Detection
2. Keypoint Detection
3. Panoptic Segmentation
#>
param(
    [string] $Dest = "Image Data",
    [switch] $DownloadImages
)

Set-StrictMode -Version Latest

Write-Host "Project tasks:"
Write-Host "  1) Object Detection"
Write-Host "  2) Keypoint Detection"
Write-Host "  3) Panoptic Segmentation"
Write-Host ""
if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest | Out-Null }

$base = 'http://images.cocodataset.org'
$tasks = @{
    annotations = @{ url = "$base/annotations/annotations_trainval2017.zip"; out = "$Dest\annotations_trainval2017.zip" }
    panoptic = @{ url = "$base/annotations/panoptic_annotations_trainval2017.zip"; out = "$Dest\panoptic_annotations_trainval2017.zip" }
}

if ($DownloadImages) {
    $tasks.images = @{ url = "$base/zips/train2017.zip"; out = "$Dest\train2017.zip" }
    $tasks.images_val = @{ url = "$base/zips/val2017.zip"; out = "$Dest\val2017.zip" }
    $tasks.images_test = @{ url = "$base/zips/test2017.zip"; out = "$Dest\test2017.zip" }
}

foreach ($k in $tasks.Keys) {
    $item = $tasks[$k]
    if (Test-Path $item.out) { Write-Host "Skipping existing:" $item.out; continue }
    Write-Host "Downloading $($item.url) -> $($item.out)"
    curl.exe -L --retry 3 --retry-delay 2 --fail $item.url -o $item.out
    if ($LASTEXITCODE -ne 0) { Write-Error "Download failed for $($item.url)"; continue }
    # Quick zip validation if python is available
    $py = & python -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) {
        Write-Host "Validating zip listing for" $item.out
        & python -m zipfile -l $item.out | Select-Object -First 5
    }
}

Write-Host "Downloads complete. Note: COCO test annotations are not publicly available." 

# --- Extraction / expansion steps ---
Write-Host "Checking and extracting archives..."

# Extract annotations zip if present and not already extracted
$annZip = Join-Path $Dest "annotations_trainval2017.zip"
$annOut = Join-Path $Dest "annotations"
if (Test-Path $annZip) {
    if (-not (Test-Path (Join-Path $annOut "annotations"))) {
        Write-Host "Extracting annotations -> $annOut"
        Expand-Archive -LiteralPath $annZip -DestinationPath $annOut -Force
    } else { Write-Host "Annotations already extracted." }
} else { Write-Host "Annotations zip not found: $annZip" }

# Extract panoptic outer bundle
$panZip = Join-Path $Dest "panoptic_annotations_trainval2017.zip"
$panOut = Join-Path $Dest "panoptic_annotations"
if (Test-Path $panZip) {
    if (-not (Test-Path (Join-Path $panOut "annotations"))) {
        Write-Host "Extracting panoptic bundle -> $panOut"
        Expand-Archive -LiteralPath $panZip -DestinationPath $panOut -Force
    } else { Write-Host "Panoptic bundle already extracted." }

    # extract nested panoptic zips using python zipfile if present
    $nestedTrain = Join-Path $panOut "annotations\panoptic_train2017.zip"
    $nestedVal   = Join-Path $panOut "annotations\panoptic_val2017.zip"
    $nestedTrainOut = Join-Path $panOut "annotations\panoptic_train2017"
    $nestedValOut   = Join-Path $panOut "annotations\panoptic_val2017"

    if (Test-Path $nestedTrain -and -not (Test-Path $nestedTrainOut)) {
        Write-Host "Extracting nested panoptic_train2017.zip -> $nestedTrainOut"
        python -m zipfile -e $nestedTrain $nestedTrainOut
    } elseif (Test-Path $nestedTrainOut) { Write-Host "panoptic_train2017 already extracted." }

    if (Test-Path $nestedVal -and -not (Test-Path $nestedValOut)) {
        Write-Host "Extracting nested panoptic_val2017.zip -> $nestedValOut"
        python -m zipfile -e $nestedVal $nestedValOut
    } elseif (Test-Path $nestedValOut) { Write-Host "panoptic_val2017 already extracted." }

} else { Write-Host "Panoptic zip not found: $panZip" }

# Extract images if requested (or if zips exist and user likely wants them expanded)
if ($DownloadImages) {
    $imgZips = @{"train" = "train2017.zip"; "val" = "val2017.zip"; "test" = "test2017.zip"}
    foreach ($k in $imgZips.Keys) {
        $z = Join-Path $Dest $imgZips[$k]
        $out = Join-Path $Dest ($k + "2017")
        if (Test-Path $z) {
            if (-not (Test-Path $out)) {
                Write-Host "Extracting $z -> $out"
                Expand-Archive -LiteralPath $z -DestinationPath $out -Force
            } else { Write-Host "$out already exists." }
        } else { Write-Host "Image archive not found (skipping): $z" }
    }
}

Write-Host "Extraction steps finished."
