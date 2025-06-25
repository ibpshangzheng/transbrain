Supported Atlases
========================================

We currently provide the following options. 


Human Brain Atlases
^^^^^^^^^^^^^^^^^^^^^

- `BN (Brainnetome Atlas) <https://atlas.brainnetome.org/>`_

    The naming of regions in the Brainnetome (BN) atlas are defined based on the anatomical locations from `Brodmann atlas <https://en.wikipedia.org/wiki/Brodmann_area>`_. You can check the correspondence in the BN website or in this `table <https://docs.google.com/spreadsheets/d/1dN6-CuPVPBUtfBX02TfaqatTBAp1FmLM/edit?usp=sharing&ouid=109641219031090954426&rtpof=true&sd=true>`_ to help understand. For subcortical regions, we adopted a `hybrid approach (22 ROIs) <https://github.com/ibpshangzheng/transbrain/tree/main/transbrain/atlas>`_ that integrates the Brainnetome Atlas, the `Allen Brain Atlas <https://community.brain-map.org/t/allen-human-reference-atlas-3d-2020-new/405>`_, and `public manual delineations <https://www.sciencedirect.com/science/article/abs/pii/S1053811913001237?via%3Dihub>`_.

- `DK (Desikan-Killiany Atlas) <https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation>`_
- `AAL (Automated Anatomical Labeling) <https://www.gin.cnrs.fr/en/tools/aal/>`_


Mouse Brain Atlas
^^^^^^^^^^^^^^^^^^^^^

- `CCFv3 (Allen Mouse Common Coordinate Framework v3) <https://atlas.brain-map.org/>`_

~~~~

- We provide several example data table in the ``exampledata`` directory of the our `GitHub repository <https://github.com/ibpshangzheng/transbrain/tree/main/transbrain/exampledata>`_.
- You can download and check these files to learn the required input format. **Note** that when replacing with your own data, please **strictly follow** the format and region order in the provided template file. Mapping will **fail** if the structure is incorrect.
- For detailed atlases information, please refer to our `preprint <https://www.biorxiv.org/content/10.1101/2025.01.27.635016v1>`_ and `transbrain/atlas <https://github.com/ibpshangzheng/transbrain/tree/main/transbrain/atlas>`_
- Support for additional atlases will be expanded in future updates.

