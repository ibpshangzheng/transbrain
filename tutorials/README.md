# Files and Analysis

## Files

* [tr_embeddings](./tr_embeddings): the region-specific transcriptional embeddings of detached model.

  * [MixData_Cortical_Train_Repeat1000](./tr_embeddings/FinalModels/MixData_Cortical_Train_Repeat1000): cortical embeddings of the integrated data repeated 100 times, with each dataset subjected to 10 iterations of model training.

  * [MixData_SubCortical_Train_Repeat1000](./tr_embeddings/FinalModels/MixData_SubCortical_Train_Repeat1000): subcortical embeddings of the integrated data repeated 100 times, with each dataset subjected to 10 iterations of model training.
 

* [structural_connection](./structural_connection): human DTI matrix and mouse tracer matrix used in our study.

## Analysis

* [tr_embeddings_analysis](./notebooks/tr_embeddings_analysis): cross-species correspondence of TR embeddings and conserved transcriptional gradients.

* [graphembeddings_analysis](./notebooks/graphembeddings_analysis): evaluate whether graph embeddings incorporate structural connectivity information while preserving cross-species transcriptional similarity.

* [translation](./notebooks/translation): three cases of applying TransBrain for cross-species translation and comparison.

    * [gradient_spectrum](./notebooks/translation/gradient_spectrum): inferring the conservation of resting fmri gradients across species.

    * [optogenetic_annotation](./notebooks/translation/optogenetic_annotation): annotating the optogenetic circuits in mice using Neurosynth.
 
    * [autism_mutation](./notebooks/translation/autism_mutation): linking gene mutations to imaging phenotype deviations in autism.
  
