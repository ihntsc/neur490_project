# neur490_project
Contains code and sample results for Katja Brand’s NEUR490 Project

## Description of contents
A description of the files that can be found in this repository and instructions on how to run any code files.
### Model code
This folder contains the code used to run the model. There are two main variations of the model - with and without unlearning.

[model.py](model%20code/model.py) - this file contains the code used to run the model without unlearning. This program can be run using job.sh, or from the command line, where it takes three arguments: the number of sleep cycles, the frequency of noise bursts, and the amplitude of noise bursts, in that order. Other parameters such as learning coefficients or number of training patterns must be changed within the code itself.

[job.sh](model%20code/job.sh) - this shell script was used to run model.py multiple times using different values of sleep cycles, noise burst frequency and amplitude. This version is configured to take one command line argument for the number of sleep cycles, then iterate over several values for both noise burst frequency and amplitude, with 10 trials for each combination of parameters. All output is written into text files corresponding to the current parameters for that set of trials.

[model_ul.py](model%20code/model_ul.py) - this file contains another version of the model, similar to model.py except with an implementation of unlearning. This program can be run using job_ul.sh, or from the command line, where it takes four arguments: the number of sleep cycles, the number of unlearning cycles, the frequency of noise bursts, and the amplitude of noise bursts, in that order.

[ul_job.sh](model%20code/ul_job.sh)  - this shell script was used to run model_ul.py. It works in a similar manner to job.sh described above. This version is configured to take two command line arguments: the number of sleep cycles, followed by the number of unlearning cycles.

### Raw data
Contains the data files obtained from many different runs of the model with different conditions.
