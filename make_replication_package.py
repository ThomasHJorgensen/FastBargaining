"""Build the replication package for the Econometric Society Data Editor.

Run from the repository root after a complete run of run_all.ipynb:

    .venv\\Scripts\\python make_replication_package.py

The script
  1. renders README.md to README.pdf (requires pandoc and Microsoft Edge; install pandoc with
     `uv sync --group build`),
  2. copies all files tracked by git, plus the files in EXTRA_FILES and minus EXCLUDE,
     to dist/Replication package/, and
  3. zips that folder to dist/Replication package.zip.

It warns if the working tree has uncommitted changes, if an exhibit listed in OUTPUT_FILES
is missing, or if the zip file exceeds the 100MB upload limit of Editorial Express.
This script is a tool for the authors and is not part of the package.
"""
import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / 'dist'
PACKAGE_NAME = 'Replication package'

# files needed by replicators that are not tracked by git
EXTRA_FILES = [
    'README.pdf',
    'LICENSE.txt',
    'DEVELOPMENT.md',
    '.python-version',
    'Application/cppfuncs/nlopt-2.4.2-dll64/libnlopt-0.lib',
]

# tracked files and folders that are not part of the package
EXCLUDE = [
    'Archive/',  # earlier implementations, not used
    '.gitignore',
    'make_replication_package.py',
]

# exhibits in the paper and the Supplemental Material (see the list of tables in README.md)
OUTPUT_FILES = [
    'Illustration/output/illustration_iEGM_interp_subplot_1.pdf',  # Figure 1 (a)
    'Illustration/output/illustration_iEGM_interp_subplot_2.pdf',  # Figure 1 (b)
    'Illustration/output/MC_unitary.tex',  # Table 2
    'Illustration/output/MC_unitary_pre.tex',  # Table 3
    'Application/output/application_iEGM_interp_subplot_1.pdf',  # Figure 2 (a)
    'Application/output/application_iEGM_interp_subplot_2.pdf',  # Figure 2 (b)
    'Application/output/MC_limited.tex',  # Table 4
    'Application/output/share_power_updated.txt',  # footnote 25
    'Application/output/calibration_fit.tex',  # Table 6
    'Application/output/calibration_parameters.tex',  # Table 7
    *[f'Application/output/shock_Klow_subplot_{i}.pdf' for i in range(1, 5)],  # Figure 3
    *[f'Application/output/shock_Khigh_subplot_{i}.pdf' for i in range(1, 5)],  # Figure 4
    'Application/output/consumption_inequality.tex',  # Table 8
    'Appendix/output/showcase_interpolation.pdf',  # Figure S1
    'Application/par_files/calibrated_par.pkl',
    'Application/solve.dll',
]

MAX_ZIP_MB = 100.0

README_CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Calibri, Arial, sans-serif; font-size: 10.5pt; line-height: 1.35; }
h1 { font-size: 17pt; } h2 { font-size: 14pt; margin-top: 1.4em; } h3 { font-size: 12pt; } h4 { font-size: 11pt; }
code { font-family: Consolas, monospace; font-size: 9pt; overflow-wrap: anywhere; }
pre { background: #f4f4f4; padding: 6px 8px; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 0.6em 0; }
th, td { border: 1px solid #999; padding: 3px 5px; vertical-align: top; text-align: left; }
th { background: #eee; }
ul.task-list { list-style: none; padding-left: 1em; }
a { color: #1a4f8b; overflow-wrap: anywhere; }
"""


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def find_edge():
    for path in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                 r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'):
        if os.path.isfile(path): return path
    return shutil.which('msedge')


def find_pandoc():
    # pandoc bundled with pypandoc_binary (uv sync --group build)
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except (ImportError, OSError):
        pass
    path = shutil.which('pandoc')
    if path: return path
    # pandoc's own installer and Anaconda's base environment, which may not be on PATH
    home = Path(os.environ.get('USERPROFILE', Path.home()))
    local = Path(os.environ.get('LOCALAPPDATA', home / 'AppData' / 'Local'))
    for path in (local / 'Pandoc' / 'pandoc.exe', Path(r'C:\Program Files\Pandoc\pandoc.exe'),
                 local / 'anaconda3' / 'Scripts' / 'pandoc.exe', home / 'anaconda3' / 'Scripts' / 'pandoc.exe',
                 local / 'miniconda3' / 'Scripts' / 'pandoc.exe', home / 'miniconda3' / 'Scripts' / 'pandoc.exe'):
        if path.is_file(): return str(path)
    return None


def build_readme_pdf():
    """ README.md -> README.html (pandoc) -> README.pdf (Edge, headless) """

    pandoc = find_pandoc()
    edge = find_edge()
    if pandoc is None: sys.exit('pandoc is needed to build README.pdf: run "uv sync --group build"')
    if edge is None: sys.exit('Microsoft Edge is needed to build README.pdf')

    tmp = DIST / 'readme_build'
    tmp.mkdir(parents=True, exist_ok=True)
    css = tmp / 'readme.css'
    css.write_text(README_CSS, encoding='utf-8')
    html = tmp / 'README.html'
    subprocess.run([pandoc, str(ROOT / 'README.md'), '-f', 'gfm', '-t', 'html5', '-s',
                    '--metadata', 'pagetitle=README', '--css', css.name, '-o', str(html)], check=True)

    pdf = ROOT / 'README.pdf'
    subprocess.run([edge, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                    f'--user-data-dir={tmp / "edge_profile"}', f'--print-to-pdf={pdf}', html.as_uri()],
                   check=True, capture_output=True)
    if not pdf.is_file(): sys.exit('README.pdf was not created')
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'built {pdf.name}')


def package_files():
    tracked = git('ls-files', '-z').split('\0')
    files = sorted(set(f for f in tracked + EXTRA_FILES if f))
    files = [f for f in files if not any(f.startswith(ex) if ex.endswith('/') else f == ex for ex in EXCLUDE)]
    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing: sys.exit('missing files:\n  ' + '\n  '.join(missing))
    return files


def main():
    allow_dirty = '--allow-dirty' in sys.argv

    # a. checks
    dirty = git('status', '--porcelain', '--untracked-files=no').strip()
    if dirty:
        msg = 'the working tree has uncommitted changes:\n' + dirty
        if not allow_dirty: sys.exit(msg + '\ncommit them first or pass --allow-dirty')
        print('WARNING: ' + msg)

    missing_output = [f for f in OUTPUT_FILES if not (ROOT / f).is_file()]
    if missing_output:
        print('WARNING: exhibits not found (run run_all.ipynb first):\n  ' + '\n  '.join(missing_output))

    # b. README.pdf
    build_readme_pdf()

    # c. copy files
    target = DIST / PACKAGE_NAME
    if target.exists(): shutil.rmtree(target)
    files = package_files()
    for f in files:
        dst = target / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst)

    # d. zip (the zip contains the folder "Replication package")
    zip_path = DIST / f'{PACKAGE_NAME}.zip'
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            zf.write(target / f, arcname=f'{PACKAGE_NAME}/{f}')

    size_mb = zip_path.stat().st_size / 1e6
    print(f'{len(files)} files copied to {target}')
    print(f'{zip_path} ({size_mb:.1f} MB)')
    if size_mb > MAX_ZIP_MB:
        print(f'WARNING: the zip file exceeds {MAX_ZIP_MB:.0f} MB, the upload limit of Editorial Express')


if __name__ == '__main__':
    main()
