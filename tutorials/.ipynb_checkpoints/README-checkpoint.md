# Files and Analysis

## Files

* [TR_embeddings](./TR_embeddings): the region-specific transcriptional embeddings of detached model.

  * [MixData_Cortical_Train_Repeat1000](./TRembeddings/FinalModels/MixData_Cortical_Train_Repeat1000): cortical embeddings of the integrated data repeated 100 times, with each dataset subjected to 10 iterations of model training.

  * [MixData_SubCortical_Train_Repeat1000](./TRembeddings/FinalModels/MixData_SubCortical_Train_Repeat1000): subcortical embeddings of the integrated data repeated 100 times, with each dataset subjected to 10 iterations of model training.
 
* [Generate_graph_embeddings](./Generate_graph_embeddings): graph embeddings of Human-Mouse generated from graph walk embedding algorithms.
 
* [Atlas](./Atlas): human and mouse atlas, templates, and corresponding csv files.

* [Structural_connection](./Structural_connection): human dti matrix and mouse tracer matrix used in our study.

## Analysis

* [TRembeddings analysis](./Notebooks/TRembeddings_analysis): cross-species correspondence of TR embeddings and conserved transcriptional gradients.

* [Graphembeddings_analysis](./Notebooks/Graphembeddings_analysis): evaluate whether graph embeddings incorporate structural connectivity information while preserving cross-species transcriptional similarity.

* [Translation](./Notebooks/Translation): three cases of applying TransBrain for cross-species translation and comparison.

    * [Gradient_spectrum](./Notebooks/Translation/Gradient_spectrum): inferring the conservation of resting fmri gradients across species.

    * [Optogenetic_annotation](./Notebooks/Translation/Optogenetic_annotation): annotating the optogenetic circuits in mice using Neurosynth.
 
    * [Autism_mutation](./Notebooks/Translation/Autism_mutation): linking gene mutations to imaging phenotype deviations in autism.
  