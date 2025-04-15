## Introduction



This folder contains all the supplementary materials for the submission (numbered 5335), titled *Linear Streaming Bandit: Regret Minimization and Fixed-Budget Epsilon-Best Arm Identification*, to the AAAI 2025 Conference Main Technical Track. The supplementary materials include:

- **Technical Appendix.pdf**. A technical appendix for the main paper. Both the missing proofs for the theoretical results and the additional experimental results are in this file.
- Python files of all the experiments mentioned in Technical Appendix.pdf.
- **data**. This folder includes Python files for data processing. 



## Guidance 



The experiments on synthetic datasets do not require loading data. Simply run them in Python:

- **experiment_CRMPS_with_baselines.py**. The regret minimization experiment on a synthethic dataset. 
- **experiment_GMPSE.py**. The Epsilon-best arm identification experiment on a synthetic dataset.
- **experiment_SPC.py**. The best arm identification experiment on a synthetic dataset.

Before running the experiments on real-world dataset, please download the Kaggle dataset [Employee’s Performance for HR Analytics]([Employee’s Performance for HR Analytics (kaggle.com)](https://www.kaggle.com/datasets/sanjanchaudhari/employees-performance-for-hr-analytics/data)) into the data folder. In this folder:

- **data_processing.py**. This script cleans the data by removing profiles with empty entries. It outputs npy files containing profiles and labels for later use.
- **data_linear_regression.py**. This script loads the profiles and labels to estimate the ground truth parameter vector $\theta^*$ for the dataset.

After modifying the corresponding load paths, these files can be run in Python:

- **realworld_CRMPS_with_baselines.py**. The regret minimization experiment on the Kaggle dataset. 
- **realworld_GMPSE.py**. The Epsilon-best arm identification experiment on the Kaggle dataset.
- **realworld_SPC.py**. The best arm identification experiment on the Kaggle dataset.



