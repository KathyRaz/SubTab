# SubTub
Data scientists frequently examine the raw content of large tables when exploring an unknown dataset. In such cases, small subsets of the full tables (sub-tables) that accurately capture table contents are useful.
We present a framework which, given a large data table T, creates a sub-table of small, fixed dimensions by selecting a subset of T's rows and projecting them over a subset of T's columns. The question is: Which rows and columns should be selected to yield an informative sub-table?

Our first contribution is an informativeness metric for sub-tables with two complementary dimensions: cell coverage, which measures how well the sub-table captures prominent data patterns in T,  and diversity.
We use association rules as the patterns captured by sub-tables, and show that computing optimal sub-tables directly using this metric
is infeasible. We then develop an efficient algorithm that indirectly accounts for association rules using table embedding. The resulting framework produces sub-tables for the full table as well as for the results of queries over the table, enabling the user to quickly understand results and determine subsequent queries. Experimental results show that high-quality sub-tables can be efficiently computed,and verify the soundness of our metrics as well as the usefulness of selected sub-tables through user studies.


The implemention of the following [paper](https://ieeexplore.ieee.org/document/10184535). 

Please cite this work as follows:

@inproceedings{amsterdamer2023selecting,
  title={Selecting Sub-tables for Data Exploration},
  author={Amsterdamer, Yael and Davidson, Susan B and Milo, Tova and Razmadze, Kathy and Somech, Amit},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={2496--2509},
  year={2023},
  organization={IEEE}
}
An example of the usage is available in the file example.ipynb
