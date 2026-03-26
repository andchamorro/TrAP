# TrAP Conda Package

This directory contains the conda recipe for building the TrAP package.

## Building the Package Locally

### Prerequisites

Install conda-build:
```bash
conda install -y conda-build conda-verify boa
```

### Build Commands

**Standard build:**
```bash
# From repository root
conda mambabuild recipe/
```

**Build with specific Python version:**
```bash
conda mambabuild recipe/ --python=3.11
```

**Build and output to specific directory:**
```bash
conda mambabuild recipe/ --output-folder dist/
```

## Testing the Built Package

After building, install from the local build:

```bash
# Find the package path
PACKAGE_PATH=$(conda build recipe/ --output)

# Create test environment and install
conda create -n trap-test -c local -c conda-forge -c bioconda trap
conda activate trap-test

# Test CLI commands
trap-train --help
trap-predict --help
trap-postprocess --help

# Test Python imports
python -c "import trap; from trap.utils.kmer import kmer_split"
```

## Linting the Recipe

Before submitting to conda-forge or bioconda:

```bash
# Check recipe syntax and best practices
conda-build recipe/ --check

# For bioconda submission
bioconda-utils lint recipe/ --packages trap
```

## Updating the Recipe

When updating the package version:

1. Update `version` in `recipe/meta.yaml`
2. If adding/removing dependencies, update the `requirements` section
3. Reset `build: number` to 0
4. Update tests if needed
5. Build and test locally before committing

## Recipe Structure

- `meta.yaml` - Main recipe file with package metadata, dependencies, and tests
- `build.sh` - Build script for Unix/macOS
- `build.bat` - Build script for Windows
- `run_test.sh` - Additional tests run during package testing

## Dependencies

### Runtime Dependencies

All runtime dependencies are specified in `meta.yaml` under `requirements: run:`. These are automatically installed when users install the trap package.

Key dependencies:
- Python >=3.9,<3.13
- PyTorch >=2.0
- Transformers >=4.30
- BioPython >=1.81
- See `meta.yaml` for complete list

### Development Dependencies

For development work, use `environment-dev.yml` which includes:
- Pinned versions for reproducibility
- conda-build tools
- Testing tools (pytest, coverage)
- Linting tools (black, flake8, isort)

## CI/CD

The package is automatically built and tested via GitHub Actions (`.github/workflows/conda-build.yml`):

- Tests run on Linux, macOS, and Windows
- Multiple Python versions tested (3.9-3.12)
- Conda package built and uploaded as artifact
- Installation from built package verified

## Publishing

### To conda-forge

1. Fork https://github.com/conda-forge/staged-recipes
2. Add recipe to `recipes/trap/`
3. Submit pull request
4. Address reviewer feedback

### To bioconda

1. Fork https://github.com/bioconda/bioconda-recipes
2. Add recipe to `recipes/trap/`
3. Submit pull request
4. Recipe automatically built and tested via CI

## Troubleshooting

**Build fails with missing dependencies:**
- Check all dependencies are available in conda-forge or bioconda
- Update channel priority: `conda config --set channel_priority strict`

**Tests fail during build:**
- Run tests locally: `pytest tests/ -v`
- Check `run_test.sh` for test commands
- Verify all test dependencies in `test: requires:`

**Package won't install:**
- Check for dependency conflicts: `conda search trap --info`
- Try with explicit channels: `conda install -c local -c conda-forge trap`

## References

- [Conda Build Documentation](https://docs.conda.io/projects/conda-build/en/latest/)
- [Conda-Forge Guidelines](https://conda-forge.org/docs/maintainer/guidelines.html)
- [Bioconda Guidelines](https://bioconda.github.io/contributor/guidelines.html)
