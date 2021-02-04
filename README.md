[![BCH compliance](https://bettercodehub.com/edge/badge/szymon319/overlapping-spheres?branch=main)](https://bettercodehub.com/)
[![codecov](https://codecov.io/gh/szymon319/overlapping-spheres/branch/main/graph/badge.svg?token=KAKJEUE2TL)](https://codecov.io/gh/szymon319/overlapping-spheres)
[![Documentation Status](https://readthedocs.org/projects/overlapping-spheres/badge/?version=latest)](https://overlapping-spheres.readthedocs.io/en/latest/?badge=latest)
![run os-tests](https://github.com/szymon319/overlapping-spheres/workflows/run%20os-tests/badge.svg)
![run unit tests](https://github.com/szymon319/overlapping-spheres/workflows/run%20unit%20tests/badge.svg)

# Structured project in mathematical modelling and numerical computation

This project contains a small Python project. We are going to use free cloud services to automate:

- unit testing on multiple Python versions
- unit testing on multiple operating systems
- coverage testing
- static analysis
- documentation generation

To make sure all dependencies are installed, we recommend creating a new virtual environment.
From the directory containing this file:

```bash
python3 -m pip install --user virtualenv
python3 -m venv venv
```

Activate the virtual environment:

Linux / macOS:
```bash
source venv/bin/activate
```

Windows cmd.exe:
```bash
venv\Scripts\activate.bat
```

Windows PowerShell:
```bash
venv\Scripts\Activate.ps1
```

Windows using Git Bash:
```bash
source venv\Scripts\activate
```

Upgrade the build tools and install this project:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .[dev,docs]
```
