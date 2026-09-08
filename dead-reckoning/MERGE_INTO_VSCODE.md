# Merge the live IDR layer into your VS Code clone

This bundle is designed to be copied into the root of the existing
`Alpha1816311/dead-reckoning` repository. It does not replace the supplied
batch workflow. `main.py`, `alignment.py`, `gnss_ins_fusion.py`, and
`map_matching.py` remain the existing pipeline; `app.py` and
`navigation_engine.py` add the incremental live path on top of those modules.

## Recommended: copy the bundle into the existing clone

1. Make a safety branch in the local VS Code terminal:

   ```bash
   git switch -c idr-live-integration
   git status
   ```

   Do not run `git reset --hard` or delete your current changes.

2. Extract this bundle **into the repository root**, preserving folders.
   When Windows asks about the existing `app.py`, choose **Replace**. This
   step is required: the checkpoint branch has a placeholder `app.py` that
   does not expose the live `engine`; keeping it causes
   `ImportError: cannot import name 'engine' from 'app'`.
   The bundle's `app.py` is the FastAPI gateway and keeps the batch work in
   `main.py`. The bundle contains the live layer, Android collector,
   dashboard, tests, model, and documentation.

   If Explorer does not show the replace prompt, extract to a temporary
   folder and run this from the repository root:

   ```powershell
   Copy-Item .\path\to\extracted-bundle\app.py .\app.py -Force
   ```

   Confirm the replacement worked:

   ```powershell
   Select-String -Path .\app.py -Pattern "engine = NavigationEngine"
   ```

   It must print one matching line.

3. Install dependencies and verify:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m pytest -q
   ```

4. Start the backend:

   ```bash
   python -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. Open `http://127.0.0.1:8000/`. Press **Run full outage demo**. The
   dashboard sends real packets to the live FastAPI engine and shows GNSS,
   outage/dead-reckoning, recovery, and trajectory state.

## Commit and push from VS Code

Only after the tests pass:

```bash
git add ARCHITECTURE.md README_LIVE.md MERGE_INTO_VSCODE.md \
  android app.py demo_replay.py models navigation_engine.py \
  requirements.txt sensor_processing.py speed_model.py tests \
  train_speed_model.py web
git commit -m "Add live IDR gateway dashboard and Android integration"
git push -u origin idr-live-integration
```

Then create a GitHub Pull Request from `idr-live-integration` into `main`.
Review the file list before merging. Keep the original batch files and
existing data changes unless your team intentionally changes them.

## If a conflict appears

Do not choose “ours” or “theirs” blindly. The live layer intentionally calls
the existing alignment and map modules, so preserve the existing function
names/imports in those files. Run `python -m pytest -q` again after resolving
any conflict.

## Android

Open the extracted `android/` folder in Android Studio, run the app on a
physical phone, and set the backend URL to the laptop's LAN address, such as
`http://192.168.1.42:8000`. The phone and laptop must be on the same Wi-Fi.
Do not use `localhost` on the phone.