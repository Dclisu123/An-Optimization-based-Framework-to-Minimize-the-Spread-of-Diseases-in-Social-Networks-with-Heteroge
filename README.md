
# An Optimization-based Framework to Minimize the Spread of Diseases in Social Networks with Heterogeneous Nodes

This repository contains the source code for the paper "An Optimization-based Framework to Minimize the Spread of Diseases in Social Networks with Heterogeneous Nodes". The files include both .py files (Python) and .m files (Matlab).

## Installation

1.  Clone the repository.
2.  Install the required dependencies using `pip`:



`pip install numpy networkx EoN matplotlib scipy pandas` 

## Usage

### Figure 1 in the paper

To display a simple graph for the network structure, run:


`python Code/draw_graph_structure.py` 

### Figure 2 in the paper

To evaluate the performance, run:


`python Code/Performance_evaluate.py --g arg` 

Here, `arg` is the argument for the network structure, where:

-   `0` is for complete,
-   `1` is for uniform random,
-   `2` is for caveman, and
-   `3` is for windmill.

### Figure 3 in the paper

To simulate the topology, run:


`python Code/simulation_topology.py --g arg1 --r0 arg2` 

Here, `arg1` is the argument for the network structure and `arg2` is the argument for R0 values, where:

-   `0` is for complete,
-   `1` is for uniform random,
-   `2` is for caveman, and
-   `3` is for windmill.

For `arg2`, `0` is for 3.8, `1` is for 5.7, and `2` is for 8.9.

### Table 3 in the appendix

To generate Table 3 in the appendix, run:


`python Code/Max_kcut_table.py --g arg1 --M arg2` 

Here, `arg1` is the argument for the network structure and `arg2` is the maximum allowable social groups. In the paper, `M` is set to `2`, `5`, and `10`.

### Figure 1 in the appendix

To simulate the topology robustness, run:


`python Code/simulation_topology_robust.py --g arg1 --p arg2` 

Here, `arg1` is the argument for the network structure and `arg2` is the failure rate, where:

-   `0` is for 0.1,
-   `1` is for 0.2, and
-   `2` is for 0.3.

### Figures 3 and 4 in the appendix

To generate Figures 3 and 4 in the appendix, run the `test_ppe.m` file under `code/Matlab`.

## Contributing

Contributions to the project are welcome. To contribute, please fork the repository, create a new branch, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contact

If you have any questions or issues, please feel free to contact us at [dclisu@tamu.edu](mailto:dclisu@tamu.edu).

## Acknowledgments

We would like to thank Dr. Hrayer Aprahamian for their help and support in this project.
