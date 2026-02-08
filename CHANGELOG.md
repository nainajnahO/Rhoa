# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Strategy framework implementation
- Preprocessing utilities
- Additional visualization options
- Performance optimizations

## [0.1.8] - 2026-02-08

### Added
- Comprehensive documentation structure
- User guide section with detailed guides for indicators, targets, and visualization
- FAQ and troubleshooting guides
- Test suite for targets module
- Automated PyPI release workflow via GitHub Actions
- RELEASE.md with complete release process documentation

### Changed
- Reorganized examples into separate pages
- Enhanced API reference documentation
- Improved docstring standards across all modules
- Removed experimental prediction models directory
- Updated CHANGELOG with real PyPI release dates

### Fixed
- Version synchronization between conf.py and pyproject.toml
- Documentation build issues

## [0.1.7] - 2025-08-11

### Added
- Additional validation in target generation
- Enhanced plot customization options

### Changed
- Improved error messages in targets module

### Fixed
- Edge cases in target generation

## [0.1.6] - 2025-08-11

### Added
- Visualization module (plots.py)
- Signal plotting with confusion matrices
- Professional chart styling

## [0.1.5] - 2025-08-11

### Added
- Target generation module with auto and manual modes
- Pareto optimization for target selection
- Elbow method for threshold detection

### Changed
- Improved target generation performance

## [0.1.3] - 2025-08-03

### Added
- Additional technical indicators
- Parabolic SAR indicator
- CCI (Commodity Channel Index)
- Stochastic Oscillator
- Average True Range (ATR)
- Bollinger Bands
- MACD indicator
- ADX (Average Directional Index)
- Williams %R indicator

### Changed
- Enhanced indicator documentation
- Improved indicator performance

### Fixed
- Edge cases in RSI calculation
- MACD signal line calculation
- Window handling in moving averages

---

## Release Notes

### Pre-Alpha Status
Rhoa is currently in pre-alpha development. The API may change between versions. We recommend pinning to specific versions in production environments.

### Upgrade Notes

#### Upgrading to 0.1.7
- No breaking changes
- Documentation improvements only

[Unreleased]: https://github.com/nainajnahO/Rhoa/compare/v0.1.7...HEAD
[0.1.7]: https://github.com/nainajnahO/Rhoa/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/nainajnahO/Rhoa/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/nainajnahO/Rhoa/compare/v0.1.3...v0.1.5
[0.1.3]: https://github.com/nainajnahO/Rhoa/releases/tag/v0.1.3
