# APPAGATO

> An **AP**proximate **PA**rallel and stochastic **G**r**A**ph querying **TO**ol for biological networks.

Version: **May, 2017**

<hr />

### Description
APPAGATO APPAGATO is a stochastic and parallel algorithm to find approximate occurrences of a query network in biological networks. APPAGATO allows nodes and edges mismatches. To speed-up the querying process, APPAGATO has been also implemented in parallel to run on graphics processing units (GPUs).

APPAGATO is developed in C++ for CUDA under GNU\Linux. It requires NVIDIA GPU devices with compute capability 3.0 and above (Kepler and Maxwell architectures).

See the related scientific article for more details.

<hr />

### License
APPAGATO is distributed under the MIT license. This means that it is free for both academic and commercial use. 
Note however that some third party components in APPAGATO require that you reference certain works in scientific publications. 
You are free to link or use APPAGATO inside source code of your own program. If do so, please reference (cite) APPAGATO and this website. We appreciate bug fixes and would be happy to collaborate for improvements.

[MIT License](https://raw.githubusercontent.com/GiugnoLab/APPAGATO/master/LICENSE)

<hr />

### References 
If you have used any of the APPAGATO project software or dataset, please cite the following article:
  
    "@article{,
    title={APPAGATO: an APproximate PArallel and stochastic GrAph querying TOol for biological networks},
    author={Bonnici, Vincenzo; Busato, Federico; Micale, Giovanni; Bombieri, Nicola; Pulvirenti, Alfredo; Giugno, Rosalba},
    journal={Bioinformatics},
    volume={32},
    number={14},
    pages={2159--2166},
    year={2016},
    publisher={Oxford Univ Press}
    }"

### Graph Dataset
[Dataset Repository]()
