Installation
============

This guide provides detailed instructions for installing Rhoa in various environments.

Requirements
------------

**Python Version**
    Python 3.9 or higher

**Required Dependencies**
    - pandas >= 1.3
    - numpy >= 1.21

**Optional Dependencies**
    - kneed - For elbow method in target generation
    - paretoset - For Pareto optimization in target generation
    - scikit-learn - For confusion matrices in visualization
    - matplotlib - For plotting functionality
    - seaborn - For enhanced visualizations

Installation Methods
--------------------

Using pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install Rhoa is using pip:

.. code-block:: bash

   pip install rhoa

This will install Rhoa and its required dependencies.

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

To install with all optional dependencies:

.. code-block:: bash

   # Install with all optional features
   pip install rhoa[all]

Or install specific optional dependencies:

.. code-block:: bash

   # For target generation features
   pip install rhoa kneed paretoset

   # For visualization features
   pip install rhoa matplotlib seaborn scikit-learn

From Source
~~~~~~~~~~~

To install the latest development version from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/nainajnahO/Rhoa.git
   cd Rhoa

   # Install in editable mode
   pip install -e .

   # Or install with development dependencies
   pip install -e ".[dev,docs]"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development work, install with development dependencies:

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/Rhoa.git
   cd Rhoa

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in editable mode with dev dependencies
   pip install -e ".[dev,docs]"

This installs:
   - Testing tools (pytest, pytest-cov)
   - Code formatters (black)
   - Linters (flake8, mypy)
   - Documentation tools (sphinx, sphinx-rtd-theme)

Virtual Environments
--------------------

It's recommended to use a virtual environment to avoid dependency conflicts.

Using venv
~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python -m venv rhoa-env

   # Activate virtual environment
   # On macOS/Linux:
   source rhoa-env/bin/activate
   # On Windows:
   rhoa-env\Scripts\activate

   # Install Rhoa
   pip install rhoa

   # When done, deactivate
   deactivate

Using conda
~~~~~~~~~~~

.. code-block:: bash

   # Create conda environment
   conda create -n rhoa-env python=3.9

   # Activate environment
   conda activate rhoa-env

   # Install Rhoa
   pip install rhoa

   # When done, deactivate
   conda deactivate

Verifying Installation
----------------------

After installation, verify that Rhoa is working correctly:

.. code-block:: python

   import rhoa
   import pandas as pd

   # Check version
   print(rhoa.__version__)

   # Test basic functionality
   series = pd.Series([1, 2, 3, 4, 5])
   sma = series.indicators.sma(window_size=3)
   print(sma)

You should see output similar to:

.. code-block:: text

   0.1.7
   0    NaN
   1    NaN
   2    2.0
   3    3.0
   4    4.0
   dtype: float64

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error: No module named 'rhoa'**
   Make sure Rhoa is installed in your active Python environment:

   .. code-block:: bash

      pip list | grep rhoa

**Import Error: No module named 'pandas'**
   Install pandas:

   .. code-block:: bash

      pip install pandas

**AttributeError: 'Series' object has no attribute 'indicators'**
   Make sure you've imported rhoa before using the accessor:

   .. code-block:: python

      import rhoa  # This registers the accessor
      import pandas as pd

      # Now you can use .indicators
      series.indicators.sma()

**Version Conflicts**
   If you encounter version conflicts with pandas or numpy:

   .. code-block:: bash

      # Upgrade to compatible versions
      pip install --upgrade pandas numpy

**Windows-Specific Issues**
   If you encounter issues on Windows, ensure you have:

   - Microsoft Visual C++ 14.0 or greater (for compiling dependencies)
   - Latest pip version: ``python -m pip install --upgrade pip``

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade rhoa

To upgrade to a specific version:

.. code-block:: bash

   pip install rhoa==0.1.7

To check your current version:

.. code-block:: python

   import rhoa
   print(rhoa.__version__)

Uninstalling
------------

To remove Rhoa:

.. code-block:: bash

   pip uninstall rhoa

Platform-Specific Notes
-----------------------

macOS
~~~~~

Rhoa works out of the box on macOS with Python 3.9+. If using the system Python, consider using Homebrew Python:

.. code-block:: bash

   brew install python@3.9
   pip3.9 install rhoa

Linux
~~~~~

Most Linux distributions include Python. On Ubuntu/Debian:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install python3.9 python3-pip
   pip3 install rhoa

Windows
~~~~~~~

Install Python from `python.org <https://www.python.org/downloads/>`_, then:

.. code-block:: bash

   python -m pip install rhoa

Docker
~~~~~~

You can use Rhoa in a Docker container:

.. code-block:: dockerfile

   FROM python:3.9-slim

   WORKDIR /app

   RUN pip install rhoa pandas numpy

   COPY . .

   CMD ["python", "your_script.py"]

Build and run:

.. code-block:: bash

   docker build -t rhoa-app .
   docker run rhoa-app

Next Steps
----------

Now that you have Rhoa installed, check out:

- :doc:`quickstart` - Get started with basic usage
- :doc:`examples/index` - See practical examples
- :doc:`user_guide/index` - Learn core concepts
- :doc:`api/index` - Explore the API reference
