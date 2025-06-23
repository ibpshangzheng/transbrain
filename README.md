# TransBrain

TransBrain is an integrated computational framework for bidirectional translation of brain-wide phenotypes between humans and mice. Specifically, TransBrain provides a systematic approach for cross-species quantitative comparison and mechanistic investigation of both normal and pathological brain functions.

![TransBrain_FIG1](./figure/fig1.png)

What can TransBrain do?

1. Inferring the conservation of whole-brain phenotypes.

2. Transforming and annotating whole-brain functional circuits.

3. Linking specific mouse models with human diseases.

![TransBrain_FIG3](./figure/fig3.png)

## Further Reading

If you wish to learn more about the construction details of this method, please refer to our article: [https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1](https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1) (in preprint).


## Installation
TransBrain is on pypi: https://pypi.org/project/transbrain/

To install TransBrain as a package, run:

```sh
pip install transbrain
```

You can also create a conda environment from the environment.yml file:

* First, clone this repository,
```sh
git clone https://github.com/ibpshangzheng/transbrain.git
```

* Then, create the environment,
```sh
cd transbrain
conda env create -f environment.yml
```

* Activate the environment,
```sh
conda activate transbrain_env
```

## Python Dependencies

The project mainly depends on Python (3.8 - 3.11).

```
matplotlib==3.7.5,
matplotlib-inline==0.1.7,
nibabel==5.2.1,
nilearn==0.10.4,
numpy==1.24.4,
openpyxl==3.1.5,
pandas==2.0.3,
scikit-learn==1.3.2,
scipy==1.10.1,
seaborn==0.13.2,
six==1.17.0
tqdm==4.67.1
ipykernel==6.29.5
```
See full list in environment.yml file. 

## Unit Test

- We provide a [Python Unit Test](https://www.dataquest.io/blog/unit-tests-python/) module to check whether the installation was successful.
- Clone our repository and run this file. Make sure you are in the ​**root directory**​ (where `test_transbrain.py` is located) before running the test.

### File Structure

    transbrain-main/
    ├── transbrain/
    │ └── exampledata/
    ├── test_transbrain.py # Unit Test file
    └── tests/ # Verification files


```bash
python test_transbrain.py
```

- If you see the message ``🎉 TransBrain installed successfully!!!``, it means that TransBrain is ready to use.


## Usage Options
To accommodate the different needs of users, we provide **three levels of usage options**. You can see detailed documentation of TransBrain [here](https://transbrain.readthedocs.io/en/latest/)

### Online Mapping
- The easiest way to get started with TransBrain. We provide a code-free website  (https://transbrain.streamlit.app/), enabling users to directly upload your data for online mapping and visualization.

### General Use
- If you want to use TransBrain to map your data and have some basic knowledge of ``Python``, we provide [**Tutorial Cases**](tutorials/README.md) demonstrating how to apply TransBrain for cross-species translation and comparison, which includes:

    * Basic example of mapping mouse phenotypes to human.

    * Basic example of mapping human phenotypes to mouse.

    * Case 1: Analyzing the conservation of resting-fMRI gradients.

    * Case 2: Annotating the optogenetic circuits in mice using Neurosynth.

    * Case 3: Linking gene mutations to imaging phenotype deviations in autism.

### Advanced Use
- If you want to understand the detailed methodology of TransBrain or adapt it to your specific needs, the [construction pipeline](pipeline/README.md) of TransBrain is also publicly available in this GitHub repository.


More detailed tutorials and API references of TransBrain can be found in our [documentation](https://transbrain.readthedocs.io/en/latest/). This helps seamless integration into your existing workflows. 🚀



## License
This project is covered under the Apache 2.0 License.

## Support
For questions and comments, please file a Github issue and/or email Shangzheng Huang(huangshangzheng@ibp.ac.cn) and cc TransBrain (transbrainmapping@gmail.com).


