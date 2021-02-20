# Differential Fairness
Code implementing differential fairness (DF) metric with demonstrations on the Adult 1994 U.S. census income data from the UCI repository

## Prerequisites

* Python
* PyTorch

The code is tested on windows and linux operating systems. It should work on any other platform.

## Differential fairness (DF) measurement

* differential_fairness.py: Python code implements function for DF using empirical counts with Dirichlet smoothing
* demo_all_groups.py: Demo for measurement of DF on all intersectional groups in Adult dataset
* demo_binary_groups.py: Demo for measurement of DF on intersectional groups in Adult dataset while each protected attribute (i.e. race) can only have binary values (i.e. white and non-white)
* 'data' folder contains the Adult dataset

## Learning algorithms
* We provide "easy to understand" implementation of learning algorithms for batch and stochastic methods. Detail instructions to run batch DF and stochastic DF classifiers are provided in the "learning-algorithm-batch" and "learning-algorithm-stochastic" folders, respectively. 

## Author

* Rashidul Islam (email: islam.rashidul@umbc.edu)
* Kamrun Naher Keya (email: kkeya1@umbc.edu)

## License

The code to implement differential fairness metric is licensed under Apache License Version 2.0.

## Acknowledgments

Many part of the Adult data pre-processing was based on the "[Towards fairness in machine learning with adversarial networks](https://github.com/equialgo/fairness-in-ml)" blog post.

##  Remarks
Code for calculating our differential fairness metric is now available in the AI Fairness 360 toolkit from IBM Research! 
[AI Fairness 360 Open Source Toolkit](http://aif360.mybluemix.net/).

## Reference Papers
* Please cite the corresponding paper when using the code
1. R. Islam, K.N. Keya, S. Pan, A.D. Sarwate, and J.R. Foulds. Differential fairness. Under submission, 2021.
2. J. R. Foulds, R. Islam, K. Keya, and S. Pan. An Intersectional Definition of Fairness. 36th IEEE International Conference on Data Engineering (ICDE), 2020. [Arxiv long version](https://arxiv.org/pdf/1807.08362.pdf).
3. J. R. Foulds, R. Islam, K. Keya, and S. Pan. Differential fairness. NeurIPS 2019 Workshop on Machine Learning with Guarantees, 2019. [PDF](http://jfoulds.informationsystems.umbc.edu/papers/2019/Foulds%20(2019)%20-%20DifferentialFairness_NeurIPS_MLWG.pdf).

