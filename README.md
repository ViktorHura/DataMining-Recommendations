
# DataMining-Classification
This repository contains code for my solution of the Recommendation assignment for the Data Mining course.




## Report

The report containing the context of this assignment, details about my solution and testing results, can be found [here](https://github.com/ViktorHura/DataMining-Recommendations/blob/main/report.pdf). 
## Installation 

You need python version `3.9` or equivalent and making a virtual environment is recommended.

```
pip install -r requirements.txt
```

## Structure

### Core scripts
`data/` contains all the original as well as transformed data, the script for it is included as `parse_data.py`.

`src/association_rules/` contains the ChatGPT created and improved implementations of the Apriori algorithm.

`src/association_rules_recommender/` contains the association rule based recommenders with evaluation code.

`recommender.py` is based on itemsets from the Apriori algorithm and `recommender-ndi.py` is based on Non-Derivable-Itemsets.

These scripts generate test data in the `cache/` directory and `visualize_data.py` will create plots based on this data.

### NDI

`src/association_rules_recommender/ndi_python/` provides a python interface to the original Non Derivable Itemsets C++ implementation.

For this to work, a combiled `ndi` binary must be present in the `ndi_python/build/` directory.
A windows binary is already included and a `CMakeLists.txt` is provided to compile the binary for other platforms, just make sure that the binary is saved in the `build/` directory.