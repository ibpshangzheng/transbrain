from setuptools import setup, find_packages

setup(
    name="transbrain",  
    version="1.8.6",        
    author="Shangzheng Huang",
    author_email="huangshangzheng@ibp.ac.cn",
    description="TransBrain is an integrated computational framework for bidirectional translation of brain-wide phenotypes between humans and mice.", 
    long_description_content_type="text/markdown",
    url="https://github.com/ibpshangzheng/transbrain",  
    packages=find_packages(), 
    include_package_data=True,
    install_requires=[
        "matplotlib==3.7.5",
        "matplotlib-inline==0.1.7",
        "nibabel==5.2.1",
        "nilearn==0.10.4",
        "numpy==1.24.4",
        "openpyxl==3.1.5",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "scipy==1.10.1",
        "seaborn==0.13.2",
        "six==1.17.0",
        "tqdm==4.67.1",
        "ipykernel==6.29.5"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.0,<3.12",
)