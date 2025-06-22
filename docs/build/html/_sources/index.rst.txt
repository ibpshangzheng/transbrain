

Welcome to TransBrain
====================================
|DOI| |Github| |PyPI| |PyPI_downloads|

.. |DOI| image:: https://img.shields.io/badge/DOI-10.1101%2F2025.01.27.635016-blue
   :target: https://doi.org/10.1101/2025.01.27.635016
   :alt: View DOI
.. |Github| image:: https://img.shields.io/badge/View_in_Github-6C51D4?logo=github&logoColor=white
   :target: https://github.com/ibpshangzheng/transbrain
.. |PyPI| image:: https://img.shields.io/pypi/v/transbrain
   :target: https://pypi.org/project/TransBrain/
.. |PyPI_downloads| image:: https://static.pepy.tech/badge/transbrain
   :target: https://pepy.tech/project/transbrain


TransBrain is an integrated computational framework for bidirectional translation of brain-wide phenotypes between humans and mice. Specifically, TransBrain provides a systematic approach for cross-species quantitative comparison and mechanistic investigation of both normal and pathological brain functions.

.. image:: ../../figure/fig1.png
    :align: center

~~~~ 

Key Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**🧬 Spatial Transcriptomic Matching**

- We integrated complementary human transcriptomic datasets, including `microarray data <http://human.brain-map.org>`_ and large-scale `single-nucleus RNA sequencing data <https://github.com/linnarsson-lab/adult-human-brain?tab=readme-ov-file>`_.  
- A detached deep neural network model was trained on the integrated transcriptomic data to learn region-specific latent embeddings that are generalizable across species.

**🧠 Graph-based Random Walk**  

- A heterogeneous graph was constructed to connect brain regions within and across species.  
- Intra-species edges were defined based on anatomical connectivity, using `viral tracer data <https://connectivity.brain-map.org/>`_ for mouse and diffusion-based tractography for human from `Human Connectome Project (HCP) <https://www.humanconnectome.org/study/hcp-young-adult/data-releases>`_.  
- Cross-species edges were determined by transcriptomic embeddings learned from the first phase, constrained by coarse-scale anatomical hierarchies.  
- Establishing a unified cross-species latent space incorporating anatomical hierarchies and structural connectivity.

**🔄 Bidirectional Mapping**  

- `Dual-regression <./methods/mapping.rst>`_ method was employed to map phenotypes cross species, using the latent embeddings defined previously.


.. image:: ../../figure/fig2.png
    :align: center

~~~~ 

Method Glossory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The process of building TransBrain involves several steps, which may be confusing for those who are not from related fields. To help users better understand the methodology, we set up a :ref:`glossory` page.

~~~~ 

What can TransBrain do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**TransBrain** offers many promising applications. You can learn how to implement these functions in the :ref:`tutorial-section` section of this documentation or through our `GitHub repository <https://github.com/ibpshangzheng/transbrain>`_.

1. Inferring the conservation of whole-brain phenotypes.
2. Transforming and annotating whole-brain functional circuits.
3. Linking specific mouse models with human diseases.

.. image:: ../../figure/fig3.png
    :align: center

~~~~ 

Usage Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- To accommodate the different needs of users, we provide **three levels of usage options**. See details in :ref:`user-guide`.

Online Mapping
^^^^^^^^^^^^^^^^
- The easiest way to get started with TransBrain. We provide a code-free website  (https://transbrain.streamlit.app/), enabling users to directly upload your data for online mapping and visualization.

General Use
^^^^^^^^^^^^^^^^
- If you want to use TransBrain to map your data and have some basic knowledge of ``Python``, you can refer to our :ref:`API` and :ref:`tutorial-section` for detailed guidance.

Advanced Use
^^^^^^^^^^^^^^^^
- If you want to understand the detailed methodology of TransBrain or adapt it to your specific needs, please refer to :ref:`build_method` section.


~~~~ 

Contents
============

.. toctree::
   :maxdepth: 2
   :caption: Setup & Usage

   installation
   user
   atlas


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   ./methods/method_glossory
   online_mapping
   tutorials/notebooks/basic_usage/basic_use_mouse_to_human.ipynb
   tutorials/notebooks/basic_usage/basic_use_human_to_mouse.ipynb
   tutorials/notebooks/translation/gradient_spectrum/fMRI_gradients.ipynb
   tutorials/notebooks/translation/optogenetic_annotation/optogenetic_circuits.ipynb
   tutorials/notebooks/translation/autism_mutation/gene_mutations.ipynb
   building



.. toctree::
   :maxdepth: 2
   :caption: API

   api/external_api
   api/utils

 


.. toctree::
   :maxdepth: 2
   :caption: Other

   faq
   citing

~~~~ 

About
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- 🌐 [`GitHub Repository <https://github.com/ibpshangzheng/transbrain>`_]
- 📦 [`Install from PyPI <https://pypi.org/project/transbrain/>`_] 
- 📜 [`Our Paper <https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1>`_]
- 📧 For questions, contact the author: Shangzheng Huang (huangshangzheng@ibp.ac.cn) and cc TransBrain (transbrainmapping@gmail.com).

