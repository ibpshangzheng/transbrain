.. _mapping:

Bidirectional Mapping
========================================
This step employs dual regression to enable quantitative comparison cross species, using the latent embedding defined previously. 

- `Dual regression <https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/DualRegression.html>`_ is a method used to project group-level patterns (e.g., networks derived from Independent Component Analysis) onto individual subject data. `Previous work <https://www.nature.com/articles/s41380-021-01298-5>`_ have also employed this method to translate brain phenotypes across species using gene expression data.

- In our pipeline, first, the value of imaging phenotype in mouse brain of ROIs was first regressed by the mouse graph embedding to calculate a **β** matrix.

- Then, we used the **β** matrix dot the graph embedding matrix of human, which will output an estimate vector consistent with the ROIs number of the human brain.

- To maintain stability, this process was repeated 500 times to generate the final average vector. The same procedure was applied in reverse for human-to-mouse mapping.


~~~~ 


🔔 This process has been embedded into the ``transbrain.trans.SpeciesTrans()`` class. You can directly refer to the source code in :ref:`Mappping_API` page and go to :ref:`tutorial-section` to learn the usage.

