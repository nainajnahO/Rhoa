# Release Process for Rhoa

This document describes the automated release process for publishing new versions of Rhoa to PyPI.

## Prerequisites

### One-Time Setup: PyPI API Token

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Scroll to "API tokens" section
3. Click "Add API token"
4. Set:
   - **Token name**: `github-actions-rhoa`
   - **Scope**: Select "Project: rhoa"
5. Copy the token (starts with `pypi-`)
6. Go to your GitHub repository: https://github.com/nainajnahO/Rhoa/settings/secrets/actions
7. Click "New repository secret"
8. Set:
   - **Name**: `PYPI_API_TOKEN`
   - **Value**: Paste your PyPI token
9. Click "Add secret"

**Important:** Save the token somewhere safe - you won't be able to see it again on PyPI!

## Release Checklist

Follow these steps for each new release:

### 1. Prepare the Release

- [ ] Ensure all changes are committed and pushed
- [ ] Run tests locally: `pytest tests/`
- [ ] Review CHANGELOG.md and move items from [Unreleased] to new version
- [ ] Update version in `pyproject.toml`
- [ ] Update version in `rhoa/__init__.py` if needed
- [ ] Update version in `docs/conf.py` if needed

### 2. Update CHANGELOG.md

```markdown
## [Unreleased]

### Planned
- Future features...

## [0.1.8] - 2026-02-08

### Added
- New feature X
- New feature Y

### Changed
- Updated Z

### Fixed
- Bug fix A
```

### 3. Commit Version Bump

```bash
git add pyproject.toml rhoa/__init__.py CHANGELOG.md docs/conf.py
git commit -m "Bump version to 0.1.8"
git push origin main
```

### 4. Create and Push Git Tag

```bash
# Create annotated tag
git tag -a v0.1.8 -m "Release version 0.1.8"

# Push tag to GitHub (this triggers the workflow!)
git push origin v0.1.8
```

### 5. Verify the Release

1. **GitHub Actions**: Go to https://github.com/nainajnahO/Rhoa/actions
   - Check that "Publish to PyPI" workflow is running
   - Wait for it to complete (usually 2-3 minutes)
   - Green checkmark = success ✅

2. **PyPI**: Go to https://pypi.org/project/rhoa/
   - Verify new version appears (may take a minute)
   - Check that description and metadata look correct

3. **GitHub Releases**: Go to https://github.com/nainajnahO/Rhoa/releases
   - New release should be created automatically
   - Includes release notes and distribution files

### 6. Test the Installation

```bash
# In a fresh environment
pip install --upgrade rhoa

# Verify version
python -c "import rhoa; print(rhoa.__version__)"
```

## Quick Reference

### Version Numbering (Semantic Versioning)

- **Patch** (0.1.7 → 0.1.8): Bug fixes, documentation
- **Minor** (0.1.x → 0.2.0): New features, backward compatible
- **Major** (0.x.x → 1.0.0): Breaking changes

### Common Issues

#### Workflow doesn't trigger
- Ensure tag starts with 'v' (e.g., v0.1.8, not 0.1.8)
- Check that tag was pushed: `git push origin --tags`

#### PyPI upload fails
- Verify PYPI_API_TOKEN secret is set correctly in GitHub
- Check token hasn't expired on PyPI
- Ensure version number in pyproject.toml wasn't already released

#### Version mismatch
- Ensure version is updated in:
  - `pyproject.toml` (line 7)
  - `rhoa/__init__.py` (line 17)
  - `docs/conf.py` (release variable)
  - `CHANGELOG.md` (new section)

## Manual Release (Fallback)

If GitHub Actions is unavailable, you can release manually:

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

## Example: Complete Release Flow

```bash
# 1. Make sure you're on main and up to date
git checkout main
git pull origin main

# 2. Update version files (edit manually)
# - pyproject.toml: version = "0.1.8"
# - rhoa/__init__.py: __version__ = "0.1.8"
# - CHANGELOG.md: Add new [0.1.8] section

# 3. Commit version bump
git add pyproject.toml rhoa/__init__.py CHANGELOG.md
git commit -m "Bump version to 0.1.8"
git push origin main

# 4. Create and push tag
git tag -a v0.1.8 -m "Release version 0.1.8"
git push origin v0.1.8

# 5. Watch the magic happen! 🎉
# GitHub Actions will automatically:
# - Build the package
# - Publish to PyPI
# - Create GitHub release
```

## Rolling Back a Release

If you need to remove a bad release:

1. **PyPI**: You cannot delete releases, but you can "yank" them:
   ```bash
   pip install twine
   twine upload --skip-existing --repository pypi dist/*
   ```
   Then go to PyPI and mark the version as "yanked"

2. **GitHub**: Delete the release and tag:
   ```bash
   # Delete remote tag
   git push origin :refs/tags/v0.1.8

   # Delete local tag
   git tag -d v0.1.8
   ```

3. **Fix and re-release**: Bump to next patch version (e.g., v0.1.9)

---

## Notes

- Always test in a development environment first
- Keep CHANGELOG.md up to date with each commit
- Tag messages should be concise (e.g., "Release version 0.1.8")
- GitHub Actions logs are helpful for debugging issues
- PyPI releases are permanent (can't be deleted, only yanked)
