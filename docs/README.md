# IDR Fusion — GitHub Pages site

Hosted at: **https://rkshaan.github.io/idr**

## Structure

```
docs/
  index.html                    ← Landing page + APK download button
  _config.yml                   ← GitHub Pages config
  .well-known/
    assetlinks.json             ← Android App Links verification (DAL)
  apk/
    IDR-Fusion-SIH-MVP.apk      ← Place the built APK here before pushing
```

## Android App Link

URL: `https://rkshaan.github.io/idr`

- If app NOT installed → shows landing page with download button
- If app IS installed → Android opens IDR Fusion directly (Navigate screen)

## assetlinks.json

After building the APK, replace `PLACEHOLDER_SHA256_REPLACE_AFTER_BUILD`
in `.well-known/assetlinks.json` with the actual SHA-256 fingerprint.

Get it with:

```
keytool -list -v -keystore ~/.android/debug.keystore \
        -alias androiddebugkey -storepass android -keypass android \
        | grep "SHA256:"
```

Or use the helper script in the project root: `Get-DebugKeySHA256.ps1`
