# Get-DebugKeySHA256.ps1
# -----------------------------------------------------------------------------
# Extracts the SHA-256 fingerprint of the Android debug keystore and
# automatically patches docs/.well-known/assetlinks.json with the real value.
#
# Run this ONCE after building the APK:
#   .\Get-DebugKeySHA256.ps1
# -----------------------------------------------------------------------------

$debugKeystore = Join-Path $env:USERPROFILE ".android\debug.keystore"
$assetLinksFile = Join-Path $PSScriptRoot "docs\.well-known\assetlinks.json"

if (-not (Test-Path $debugKeystore)) {
    Write-Error "Debug keystore not found at: $debugKeystore"
    Write-Host "Build the APK first (gradlew assembleDebug) to generate the debug key."
    exit 1
}

# Find keytool — try Android Studio JBR first, then PATH
$keytool = $null
$candidates = @(
    "C:\Program Files\Android\Android Studio\jbr\bin\keytool.exe",
    "C:\Program Files\Android\Android Studio\jre\bin\keytool.exe"
)
foreach ($c in $candidates) {
    if (Test-Path $c) { $keytool = $c; break }
}
if (-not $keytool) {
    $found = Get-Command keytool -ErrorAction SilentlyContinue
    if ($found) { $keytool = $found.Source }
}
if (-not $keytool) {
    Write-Error "keytool not found. Install Android Studio or add Java/bin to PATH."
    exit 1
}

Write-Host "Using keytool: $keytool"
Write-Host "Keystore:      $debugKeystore"
Write-Host ""

# Run keytool
$output = & "$keytool" -list -v `
    -keystore "$debugKeystore" `
    -alias androiddebugkey `
    -storepass android `
    -keypass android 2>&1

# Extract SHA256 line
$sha256Line = $output | Where-Object { $_ -match "SHA256:" } | Select-Object -First 1
if (-not $sha256Line) {
    Write-Error "Could not extract SHA256 fingerprint. keytool output:`n$output"
    exit 1
}

# Parse — format: "         SHA256: AA:BB:CC:..."
$fingerprint = ($sha256Line -replace ".*SHA256:\s*", "").Trim()

Write-Host "SHA-256 Fingerprint:"
Write-Host "  $fingerprint"
Write-Host ""

# Patch assetlinks.json
if (-not (Test-Path $assetLinksFile)) {
    Write-Error "assetlinks.json not found at: $assetLinksFile"
    exit 1
}

$json = Get-Content $assetLinksFile -Raw
if ($json -match "PLACEHOLDER_SHA256_REPLACE_AFTER_BUILD") {
    $json = $json -replace "PLACEHOLDER_SHA256_REPLACE_AFTER_BUILD", $fingerprint
    Set-Content -Path $assetLinksFile -Value $json -NoNewline
    Write-Host "Patched: $assetLinksFile"
} else {
    Write-Host "assetlinks.json already contains a fingerprint — no change made."
    Write-Host "Current content:"
    Write-Host $json
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. git add docs/.well-known/assetlinks.json"
Write-Host "  2. git commit -m 'chore: update assetlinks fingerprint'"
Write-Host "  3. git push  (triggers GitHub Pages deploy)"
Write-Host "  4. Verify: https://rkshaan.github.io/.well-known/assetlinks.json"
Write-Host "  5. On device: adb shell pm verify-app-links --re-verify com.idr.mobile"
