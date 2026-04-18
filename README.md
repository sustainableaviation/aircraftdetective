# AircraftDetective

[![PyPI Downloads](https://img.shields.io/pypi/dm/aircraftdetective?label=PyPI%20Downloads&logo=pypi&logoColor=white)](https://pypistats.org/packages/aircraftdetective)
[![License: MIT](https://img.shields.io/pypi/l/aircraftdetective?label=License&logo=open-source-initiative&logoColor=white)](https://pypi.org/project/aircraftdetective/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aircraftdetective?logo=python&logoColor=white)](https://pypi.org/project/aircraftdetective/)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

A Python package for estimating the efficiency of commercial aircraft.  
Maintenance Team: [@michaelweinold](https://github.com/michaelweinold)

## Installation

See [the package documentation](https://aircraftdetective.readthedocs.io/) for installation instructions.

## Development

### Documentation

The package documentation is based on [`mkdocs`](https://www.mkdocs.org). To build the documentation locally, install the required packages from the `docs/_requirements.txt` file and navigate to the package root directory to execute:

```bash
mkdocs serve
```

### Testing

Package tests are based on [`pytest`](https://docs.pytest.org/en/stable/). To run all tests, navigate to the package root directory, install the testing dependencies, and execute:

```bash
pip install -e .[testing]
pytest
```

When developing with Visual Studio Code, tests can also be run from [the Test Explorer sidebar](https://code.visualstudio.com/docs/python/testing).

### CI/CD

The package uses [GitHub Actions](https://github.com/features/actions) for continuous integration and deployment. The CI/CD pipeline is defined in the `.github/workflows` directory.

| Workflow | Description | Trigger |
|----------|-------------|---------|
| `.github/workflows/test_package.yml` | Runs the test suite across supported Python versions. | Pull requests to `main` and `dev`, pushes to `main` and `dev`, and manual triggers. |
| `.github/workflows/publish_testpypi.yml` | Runs tests, builds distributions, and uploads the package to TestPyPI. | Every new Git tag and manual triggers. |
| `.github/workflows/publish_pypi.yml` | Runs tests, builds distributions, and uploads the package to PyPI. | Every new GitHub release. |
| `.github/workflows/update_license.yml` | Updates the copyright year(s) in the license file. | Scheduled annually on January 1. |
