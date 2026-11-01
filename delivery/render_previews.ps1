# Render every slide of a PPTX to PNG with Microsoft PowerPoint (COM), for visual review.
# Usage: powershell -File delivery\render_previews.ps1 -Deck delivery\person_c_enhancement.pptx -OutDir delivery\.build\c_previews
param(
    [Parameter(Mandatory = $true)][string]$Deck,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$Width = 1280,
    [int]$Height = 720
)
$deckPath = (Resolve-Path $Deck).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$outPath = (Resolve-Path $OutDir).Path
$ppt = New-Object -ComObject PowerPoint.Application
$alreadyOpen = $ppt.Presentations.Count
# ReadOnly, Untitled, WithWindow = false
$pres = $ppt.Presentations.Open($deckPath, $true, $false, $false)
try {
    for ($i = 1; $i -le $pres.Slides.Count; $i++) {
        $pres.Slides.Item($i).Export((Join-Path $outPath ("slide-{0:D2}.png" -f $i)), "PNG", $Width, $Height)
    }
    Write-Output "exported $($pres.Slides.Count) slides to $outPath"
} finally {
    $pres.Close()
    # Never close a PowerPoint session the user already had open.
    if ($alreadyOpen -eq 0 -and $ppt.Presentations.Count -eq 0) { $ppt.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
