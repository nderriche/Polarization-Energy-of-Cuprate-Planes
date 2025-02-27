# Topology-of-Alkali-and-Alkaline-Earth-Materials


# Charge Susceptibility Calculations for Hybridized Materials

Contains the Python scripts used in the analysis and visualization of the charge-based topological properties of materials. The values associated with parameters and functions in these scripts are tailored for calculations associated with all alkali and alkaline earth 1D and 2D elemental systems specifically, which led to the publication of the following paper showing the emergence of interesting topological edge states in the lighter elements.   
**[Link to Paper](https://arxiv.org/abs/2405.00787)**

Here is an overall breakdown of the Python scripts included in this repository. They all are structured using code cells containing clear comments explaining the purpose of each section and function, so it is useful to have a look at the contents of the files directly for more detail.

External Density Functional Theory (DFT) orbital-weighted band structure calculations for all of the alkali and alkaline earth elemental 1D chain and 2D hexagonal structures with varying lattice constants first need to be performed, as they are imported and analyzed in the following code. In my case, I have used the DFT software FPLO, and the Python-based framework Phonopy to do so.


## 1. [bobo](bobo.py)
