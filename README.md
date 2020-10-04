# neur490_project
Contains code and sample data for Katja Brand’s NEUR490 Project.

## Description of contents
A description of the files that can be found in this repository and instructions on how to run any code files.
### Model code
This folder contains the code used to run the model. There are two main variations of the model - with and without unlearning.

[model.py](model%20code/model.py) - this file contains the code used to run the model without unlearning. This program can be run using job.sh, or from the command line, where it takes three arguments: the number of sleep cycles, the frequency of noise bursts, and the amplitude of noise bursts, in that order. Other parameters such as learning coefficients or number of training patterns must be changed within the code itself.

[job.sh](model%20code/job.sh) - this shell script was used to run model.py multiple times using different values of sleep cycles, noise burst frequency and amplitude. This version is configured to take one command line argument for the number of sleep cycles, then iterate over several values for both noise burst frequency and amplitude, with 10 trials for each combination of parameters. All output is written into text files corresponding to the current parameters for that set of trials.

[model_ul.py](model%20code/model_ul.py) - this file contains another version of the model, similar to model.py except with an implementation of unlearning. This program can be run using job_ul.sh, or from the command line, where it takes four arguments: the number of sleep cycles, the number of unlearning cycles, the frequency of noise bursts, and the amplitude of noise bursts, in that order.

[ul_job.sh](model%20code/ul_job.sh)  - this shell script was used to run model_ul.py. It works in a similar manner to job.sh described above. This version is configured to take two command line arguments: the number of sleep cycles, followed by the number of unlearning cycles.

### Sample data
This folder contains the data files obtained from many different runs of the model with different conditions. Files are generally named in the convention c1f2a3.txt where the number after c represents the number of cycles, the number after f represents the frequency of noise bursts, and the number after a represents the amplitude of noise bursts.

[5_pats](sample%20code/5_pats) - this folder contains the results obtained when running the model with only five initial training patterns rather than 10. Data was obtained for both no sleep and sleep with 4800 cycles, at a variety of noise burst amplitudes and frequencies.

[altered_training_cycles](sample%20code/altered_training_cycles) - this folder contains results obtained from a variation of the regular model that allowed fewer initial and interference training cycles.

[c0](sample%20code/c0) - this folder contains results for running the standard model with zero cycles of sleep - the 'no sleep' condition. Though files are named with different frequencies and amplitudes, this was only to produce the same number of results as other conditions, there is no difference between them.

[c100](sample%20code/c100) - this folder contains results for running the standard model with 100 cycles of sleep.


[c1200](sample%20code/c1200) - this folder contains results for running the standard model with 1200 cycles of sleep.


[c2400](sample%20code/c2400) - this folder contains results for running the standard model with 2400 cycles of sleep.


[c4800](sample%20code/c4800) - this folder contains results for running the standard model with 4800 cycles of sleep.


[c9200](sample%20code/c9200) - this folder contains results for running the standard model with 9200 cycles of sleep.


[ltm=0.005](sample%20code/ltm=0.005) - this folder contains results for running a version of the standard model with 4800 cycles of sleep and a LTM learning coefficient of 0.005.

[pos_analysis](sample%20code/pos_analysis) - this folder contains the results used to conduct an analysis on the effect of pattern position on the final error rate.


[ul_v1](sample%20code/ul_v1) - this folder contains results for running the unlearning version of the model with version 1 of unlearning (4800 sleep cycles, 1200 unlearning cycles, STM coefficient = -0.01).

[ul_v2](sample%20code/ul_v2) - this folder contains results for running the unlearning version of the model with version 2 of unlearning (4800 sleep cycles, 1200 unlearning cycles, STM coefficient = -0.001).

[ul_v3](sample%20code/ul_v3) - this folder contains results for running the unlearning version of the model with version 3 of unlearning (4800 sleep cycles, 1200 unlearning cycles (occurring in between sleep cycles), STM coefficient = -0.001).

[ul_v4](sample%20code/ul_v4) - this folder contains results for running the unlearning version of the model with version 4 of unlearning (1200 sleep cycles, 1000 unlearning cycles, STM coefficient = -0.0001).
