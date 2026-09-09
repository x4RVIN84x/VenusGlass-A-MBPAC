# VG VISTA Windows Packaging

VG VISTA is packaged as a **Nuitka standalone** application and installed with
**Inno Setup**. This keeps Python and Qt private to the application rather than
requiring an IDE or Python installation on the operator workstation.

## Build workstation setup

Use 64-bit Python on a Windows build workstation, then install the project
runtime dependencies and build-only tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
```

Install Microsoft C++ Build Tools and Inno Setup 6 on the build workstation.
They are build tools only; neither is required on the target operator PC.

## Create the application and installer

From the repository root:

```powershell
.\packaging\build_vg_vista.ps1
.\packaging\build_vg_vista.ps1 -CreateInstaller
```

The first command generates `hmi_app/qml/assets/vg-vista.ico` from the approved
Venus Glass logo and writes the standalone application under
`build/vg-vista/`. The second also writes the installer to `dist/installer/`.
Pass `-InnoSetupCompiler 'C:\path\to\ISCC.exe'` when Inno Setup is installed
outside its normal location.

The build defaults to `-Compiler MSVC`, which enforces the Visual Studio 2022
toolchain recommended for a release build. `-Compiler MinGW64` is retained for
diagnostics only; do not use it for a production release of this PySide6 app.

The `.iss` installer intentionally stores the inspection SQLite database under
the operator's local application-data directory, not under `Program Files`.
This preserves history across upgrades and avoids permissions failures. For an
approved portable deployment, set `VG_VISTA_PORTABLE=1` before launch.

## Release checks

Test the installer on a clean Windows workstation with a camera and a recipe.
Verify the QML splash, icon, camera view, report persistence, and PLC remains
disabled until separately commissioned. Sign both `VG VISTA.exe` and the setup
EXE with the company code-signing certificate before distribution; no signing
certificate or PLC credentials belong in this repository.
