# An Optimization-based Framework to Minimize the Spread of Diseases in Social Networks with Heterogeneous Nodes

## Overview
This repository contains the source code for the paper "An Optimization-based Framework to Minimize the Spread of Diseases in Social Networks with Heterogeneous Nodes". 
The files are including both .py field (python) and .m file (Matlab). 

## Installation
1. Clone the repository.
2. Install the required dependencies using `pip`:\
pip install numpy\
pip install networkx\
pip install EoN\
pip install matplotlib\
pip install scipy\
pip install pandas



## Usage
1. Figure 1 in the paper:\
python draw_graph_structure.py\
It shows a simple graph for network strcuture

2. Figure 2 in the paper:\
python Performance_evaluate.py --g arg
--g is the argument for network streucutre, 0 for complete, 1 for uniform random, 2 for caveman, 3 for windmill

3. Figure 3 in the paper:\
python simulation_topology.py --g arg1 --r0 arg 2 
--g is the argument for network streucutre, 0 for complete, 1 for uniform random, 2 for caveman, 3 for windmill
--r0 is the argument for R0 values, 0 for 3.8, 1 for 5.7, 2 for 8.9

4. Table 3 in the appendix:\
python Max_kcut_table.py --g arg1 --M arg 2
--g is the argument for network streucutre, 0 for complete, 1 for uniform random, 2 for caveman, 3 for windmill
--M is the maximuam allowable social groups. In the paper we choose M = 2,5,10.

5.Figure 1 in the appendix:\
python simulation_topology_robust.py --g arg1 --p arg 2
--g is the argument for network streucutre, 0 for complete, 1 for uniform random, 2 for caveman, 3 for windmill
--p is the failure rate, 0 for 0.1, 1 for 0.2, 2 for 0.3

6.Figure 3,4 in the appendix:
'test_ppe.m' under code/Matlab

## Contributing
Contributions to the project are welcome. To contribute, please fork the repository, create a new branch, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact
If you have any questions or issues, please feel free to contact us at dclisu@tamu.edu.

## Acknowledgments
We would like to thank Dr.Hrayer Aprahamian for their help and support in this project.
