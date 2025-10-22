# FaceApp.spec  — build with:  pyinstaller --clean --noconfirm FaceApp.spec
from PyInstaller.utils.hooks import collect_all

# Your EXACT pip names -> importable module names
PIP_TO_IMPORT = {
    "aiofiles": "aiofiles",
    "altgraph": "altgraph",
    "annotated-types": "annotated_types",
    "anyio": "anyio",
    "asgiref": "asgiref",
    "Brotli": "brotli",
    "certifi": "certifi",
    "charset-normalizer": "charset_normalizer",
    "click": "click",
    "colorama": "colorama",
    "comtypes": "comtypes",
    "Django": "django",
    "face-alignment": "face_alignment",
    "fastapi": "fastapi",
    "ffmpy": "ffmpy",
    "filelock": "filelock",
    "fsspec": "fsspec",
    "gradio": "gradio",
    "gradio_client": "gradio_client",
    "groovy": "groovy",
    "h11": "h11",
    "httpcore": "httpcore",
    "httpx": "httpx",
    "huggingface-hub": "huggingface_hub",
    "idna": "idna",
    "imageio": "imageio",
    "Jinja2": "jinja2",
    "lazy_loader": "lazy_loader",
    "llvmlite": "llvmlite",
    "markdown-it-py": "markdown_it",
    "MarkupSafe": "markupsafe",
    "mdurl": "mdurl",
    "mpmath": "mpmath",
    "networkx": "networkx",
    "numba": "numba",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "orjson": "orjson",
    "packaging": "packaging",
    "pandas": "pandas",
    "pefile": "pefile",
    "pillow": "PIL",
    "pydantic": "pydantic",
    "pydantic_core": "pydantic_core",
    "pydub": "pydub",
    "Pygments": "pygments",
    "pygrabber": "pygrabber",
    "pyinstaller": "PyInstaller",
    "pyinstaller-hooks-contrib": "PyInstaller_hooks_contrib",
    "python-dateutil": "dateutil",
    "python-multipart": "multipart",
    "pytz": "pytz",
    "pywin32-ctypes": "pywin32_ctypes",
    "PyYAML": "yaml",
    "requests": "requests",
    "rich": "rich",
    "ruff": "ruff",
    "safehttpx": "safehttpx",
    "scikit-image": "skimage",
    "scipy": "scipy",
    "semantic-version": "semantic_version",
    "setuptools": "setuptools",
    "shellingham": "shellingham",
    "six": "six",
    "sniffio": "sniffio",
    "sqlparse": "sqlparse",
    "starlette": "starlette",
    "sympy": "sympy",
    "tifffile": "tifffile",
    "tomlkit": "tomlkit",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "typer": "typer",
    "typing-inspection": "typing_inspection",
    "typing_extensions": "typing_extensions",
    "tzdata": "tzdata",
    "urllib3": "urllib3",
    "uvicorn": "uvicorn",
    "websockets": "websockets",
}

def grab_all(pkgs):
    datas, bins, hidden = [], [], []
    for p in pkgs:
        try:
            d, b, h = collect_all(p)
            datas += d; bins += b; hidden += h
        except Exception:
            # if a package has no importable module or no data, just skip
            pass
    return datas, bins, hidden

datas, bins, hidden = grab_all(PIP_TO_IMPORT.values())

# Include your asset folders beside the exe so users can add models
datas += [
    ('functions/images', 'functions/images'),
    ('functions/models', 'functions/models'),
    ('pretrained', 'pretrained'),
    ('functions/seguiemj.ttf','functions/seguiemj.ttf')
]

block_cipher = None
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=bins,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    name='FaceApp', console=True, icon='app.ico',
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name='FaceApp'
)
