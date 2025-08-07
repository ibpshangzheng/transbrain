Installation
=============

Installation Options
--------------------
TransBrain is on PyPI: https://pypi.org/project/transbrain/

To install TransBrain as a package in your environment, run:

.. code-block:: bash

      pip install transbrain

To create a new Conda environment with TransBrain included, use `environment.yml <https://github.com/ibpshangzheng/transbrain/blob/main/environment.yml>`_ on Linux, or `environment_win.yml <https://github.com/ibpshangzheng/transbrain/blob/main/environment_win.yml>`_ / `environment_mac.yml <https://github.com/ibpshangzheng/transbrain/blob/main/environment_mac.yml>`_ for Windows and macOS:

- First, clone this `repository <https://github.com/ibpshangzheng/transbrain>`_ ,

.. code-block:: bash

      git clone https://github.com/ibpshangzheng/transbrain.git

- Then, create the environment,

.. code-block:: bash

      cd transbrain
      
      conda env create -f environment.yml

- Activate the environment,

.. code-block:: bash

      conda activate transbrain_env

Python Dependencies
-------------------

The project mainly depends on Python (3.8 - 3.11). Key dependencies include::

    matplotlib==3.7.5
    matplotlib-inline==0.1.7
    nibabel==5.2.1
    nilearn==0.10.4
    numpy==1.24.4
    openpyxl==3.1.5
    pandas==2.0.3
    scikit-learn==1.3.2
    scipy==1.10.1
    seaborn==0.13.2
    six==1.17.0
    tqdm==4.67.1
    ipykernel==6.29.5
    anndata==0.9.2

See the full list in the `environment.yml <https://github.com/ibpshangzheng/transbrain/blob/main/environment.yml>`_ file.


Unit Test
-------------------
- We provide a `Python Unit Test <https://www.dataquest.io/blog/unit-tests-python/>`_ module to check whether the installation was successful.
- Clone our repository and run this file. Make sure you are in the **root directory** (where `test_transbrain.py` is located) before running the test.

.. code-block:: text

    transbrain-main/
    ├── transbrain/
    │   └── exampledata
    ├── test_transbrain.py (Unit Test file)
    └── tests (Files used to verify installation)

.. code-block:: bash

      python test_transbrain.py


- If you see the message ``🎉 TransBrain installed successfully!!!``, it means that TransBrain is ready to use.
