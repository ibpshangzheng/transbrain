.. _glossory:

Method Glossory
================
To better help users understand the terms involved in the TransBrain methodology, we provide a glossary with additional links, covering: **Datasets** and **Methods or Concepts**.


Datasets
---------------------

.. glossary::

    AHBA data
        The Allen Human Brain Atlas (AHBA) is a microarray dataset of gene expression patterns in the human brain, derived from post-mortem tissue samples. It provides detailed maps of where specific genes are expressed across different regions of the brain, helping to understand brain function and structure.

        * link: http://human.brain-map.org
        * article: https://www.nature.com/articles/nature11405

    
    Human single-nucleus data
        Single-nucleus RNA sequencing (snRNA-seq) data from human brain tissue provided by the Linnarsson laboratory. This dataset includes single-nucleus transcriptomic data from multiple brain regions of adult human donors.

        * link: https://github.com/linnarsson-lab/adult-human-brain?tab=readme-ov-file
        * article: https://www.science.org/doi/abs/10.1126/science.add7046

    
    Mouse spatial transcriptomic data
        Whole-brain spatial transcriptomic (ST) sequencing data from three adult male mice from Ortiz et al.'s study.

        * link: http://molecularatlas.org
        * article: https://www.science.org/doi/full/10.1126/sciadv.abb3446


    Diffusion Weighted Imaging
        Diffusion Weighted Imaging is an advanced MRI technique that maps the diffusion of water molecules along white matter fibers in the brain. By mapping the direction and density of water diffusion, it helps identify the connections between different brain regions, allowing researchers to study the brain's structural network and its organization.


    Human Connectome Project - Young Adult (HCP-YA)
        The Human Connectome Project - Young Adult (HCP-YA) is a comprehensive neuroimaging study aimed at mapping the human brain's structural and functional connectivity in healthy young adults. Conducted by the WU-Minn Consortium, the project involved over 1,200 participants aged 22-35 and utilized advanced imaging techniques to provide detailed insights into brain organization.

        * link https://www.humanconnectome.org/study/hcp-young-adult/data-releases

    Mouse tracer data
        Tracer technology is a neuroscience method that uses molecular markers to map neural pathways by injecting a tracer into a brain region, which is then transported along neurons to track their connectivity.
        The mouse tracer data used in our project was sourced from the Allen Mouse Connectivity Atlas.

        * link https://connectivity.brain-map.org/

    Human resting-state fMRI
        Resting-state fMRI is a neuroimaging technique that measures brain activity through blood oxygen level changes while the subject is at rest. It helps understand the brain's intrinsic networks and is commonly used in studies of brain function and connectivity. The data used here were obtained from the HCP Young Adult 1200 Subjects Data Release.

        * link: https://www.humanconnectome.org/study/hcp-young-adult/data-releases

    Mouse resting-state fMRI
        The resting-state fMRI data for awake mice, made publicly available by Gozzi et al., were obtained from 10 adult male mice (aged < 6 months) using a 7T Bruker MRI scanner.

        * link: https://data.mendeley.com/datasets/np2fx99hn6/2
        * article: https://www.cell.com/current-biology/fulltext/S0960-9822(21)01691-2


    ABIDE I dataset
        The ABIDE I (Autism Brain Imaging Data Exchange I) is a large-scale collection of neuroimaging dataset aimed at understanding the neural basis of Autism Spectrum Disorder (ASD).

        * link https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html


    Mouse structural imaging dataset 
        A large set of autism mouse models, composed of separate cohorts from various laboratories, and scanned at the Mouse Imaging Centre in Toronto.

        * link: https://www.braincode.ca/content/public-data-releases#dr001.
        * article: https://www.nature.com/articles/mp201498


Methods or Concepts
---------------------

.. glossary::
    Microarray
        Microarray technology measures gene expression levels across thousands of genes simultaneously using fixed probes on a solid surface, offering high-throughput gene expression analysis.

    Single-nucleus RNA sequencing
        snRNA-seq analyzes gene expression at the individual cell level, offering insights into the molecular diversity of different brain cell types.

    Spatial transcriptomic data
        Spatial transcriptomics combines gene expression analysis with tissue spatial context, enabling the mapping of gene activity across tissue sections while preserving their spatial organization.

    Brainnetome atlas
        The Brainnetome Atlas is a human brain atlas constructed based on structural connectivity, which divides the brain into 246 regions

        * link: https://atlas.brainnetome.org/
        * article: https://www.science.org/doi/full/10.1126/sciadv.abb3446


    Desikan-Killiany atlas
        The Desikan-Killiany (DK) Atlas is a widely used brain atlas for human neuroimaging studies, particularly in structural MRI. It divides the brain into distinct regions based on anatomical landmarks. 

        * link: https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation


    AAL atlas
        Automated Anatomical Labeling (AAL) Atlas is a widely used brain atlas that provides a standardized map for labeling anatomical regions of the human brain. It divides the brain into predefined regions of interest (ROIs) based on structural MRI scans. 

        * link: https://www.gin.cnrs.fr/en/tools/aal/


    Allen Mouse Brain Atlas
        The Allen Mouse Brain Atlas CCFv3 (Common Coordinate Framework version 3) is a 3D reference brain atlas for the mouse brain. The CCFv3 space is built by aligning high-resolution brain images from multiple mice to a single reference brain, allowing for precise localization of brain regions.

        * link: https://atlas.brain-map.org/
        * article: https://www.cell.com/cell/fulltext/S0092-8674(20)30402-5?dgcid=raven_jbs_aip_email


    Cross-species homology mapping
        Cross-species homology region mapping involves identifying conserved genomic regions between different species that share similar functions or evolutionary origins.


    Neural network 
        A neural network is a computational model inspired by the way biological neural networks in the brain process information. It consists of layers of interconnected nodes (neurons) that work together to solve complex tasks such as classification, regression, and pattern recognition.
        

    Embeddings
        Embeddings represent high-dimensional data in a lower-dimensional space, while preserving important relationships and patterns. They also help capture semantic similarities, allowing for better generalization and improved performance.


    Gene enrichment analysis
        Gene enrichment analysis identifies biological functions or pathways that are overrepresented in a specific gene set, helping to reveal the underlying molecular processes or disease mechanisms.


    Human-Mouse Graph
        The Human-Mouse graph is a graph structure used to represent the similarities between brain regions in humans and mice. In this graph, brain regions from both species are represented as nodes, and the edges (connections) between nodes are weighted based on transcriptional similarity and structural connectivity. 


    Random Walk
        A random walk explores a graph to learn node embeddings based on their neighborhood structures. In TransBrain, Node2Vec algorithm was used, with the probability of moving to a neighbor depending on the graph structure and walk parameters.


    Dual regression
        Dual regression is a method used in neuroimaging to project group-level patterns (such as brain networks derived from Independent Component Analysis, ICA) onto individual subject data. Previous work have also employed this method to translate brain phenotypes across species using  gene expression data.

        * link: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/DualRegression.html
        * article: https://www.nature.com/articles/s41380-021-01298-5

    Phenotype
        In neuroscience, phenotype refers to the observable characteristics or traits of the brain, including brain structure, function, and individual behavior. Phenotypic traits can include aspects such as cognitive abilities, neural connectivity, or responses to stimuli, and they are often studied to understand the biological basis of diseases or neurological disorders.
    
    Functional gradients
        Functional gradients refer to inherent patterns of brain activity or connectivity that smoothly transition across different brain regions, reflecting the gradual shift in neural function and cognitive processes within the brain's organization.

        * article https://www.pnas.org/doi/abs/10.1073/pnas.1608282113

    Co-activation patterns (CAPs)
        Co-activation patterns (CAPs) are brain activity patterns where different regions are activated together, used to study functional brain networks and their collaboration during tasks or rest.

    Optogenetic fMRI
        Optogenetic fMRI combines optogenetics and functional MRI to control and monitor brain activity. By using light to stimulate specific neurons that have been genetically modified to express light-sensitive proteins, researchers can map the causal relationship between neural activity and brain function in real time. 

    Neurosynth
        Neurosynth is a meta-analytical platform capable of generating keyword-based statistical functional maps that display activation patterns of specific cognitive processes or psychological terms.

        * link https://neurosynth.org

    Deformation-based morphology
        Deformation-based morphology (DBM) is a technique used to study structural brain changes by analyzing the spatial deformation of brain images. It involves mapping brain structures between individual subjects and a common template to detect differences in brain volume.

    Normative model
        Normative model is a statistical method that represents typical brain structure or function in a healthy population. It serves as a baseline or reference to compare individual data, helping identify deviations that may indicate abnormalities, such as in neurological or psychiatric disorders. 