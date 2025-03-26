
# Model Training

In order to generate results there are two scripts: `world_model_learning.py` and `policy_learning_experiments.py`. Once you have trained the models, you can generate the results plot using: `code/visualize_results.ipynb`.

## World Model Learning Script:

In the paper we use only one world model, shared across all policies learned. To train your own world model run `world_model_learning.py`. There are default values for the arguments, so it can be ran directly. 

## Policy Training Job Management Script: 

The bash script `train_manager.sh` (in general) automates the process of running, listing, querying, and stopping multiple instances of a python training script. It allows for job management by running parallel experiments with different parameters while logging execution details. Although bespoke to this repo, it can be adjusted for other projects. In this project we use it to run instances of `policy_learning_experiments.py` to then aggregate statistics of multiple trainings used in the results of the paper

### Prerequisites
- The training script should be located at `scripts/policy_learning_experiments.py`.
- The script should be executed in a Unix/Linux environment.
- You may need to run in the root directory to include the python training script in your PYTHONPATH: `export PYTHONPATH=$(pwd)/code`

## Usage
The script supports multiple commands:

### 1. Start Training Jobs
```bash
./script.sh start <environment_type> <influence_mode>
```
Starts multiple training jobs (default: 5) with the given environment and influence-mode.

Example:
```bash
./script.sh start multi-agent perfect-information
```

### 2. List Running Jobs
```bash
./script.sh list <environment_type> <influence_mode>
```
Displays active jobs for the specified environment and influence mode.

Example:
```bash
./script.sh list multi-agent perfect-information
```

### 3. List All Running Jobs
```bash
./script.sh list_all
```
Shows all running jobs across all experiments.

### 4. Query a Specific Job
```bash
./script.sh query <environment_type> <influence_mode> <PID>
```
Provides detailed information about a specific job using its process ID (PID).

Example:
```bash
./script.sh query multi-agent perfect-information 12345
```

### 5. Kill a Specific Job
```bash
./script.sh kill <environment_type> <influence_mode> <PID>
```
Terminates a specific training job using its PID.

Example:
```bash
./script.sh kill multi-agent perfect-information 12345
```

### 6. Stop All Jobs for a Given Experiment
```bash
./script.sh stop <environment_type> <influence_mode>
```
Stops all running jobs for a specific environment and influence mode.

Example:
```bash
./script.sh stop multi-agent perfect-information
```

### 7. List Available Experiments
```bash
./script.sh list_experiments
```
Lists all active environment and influence mode combinations with running jobs.

### 8. Clean and Archive Logs
```bash
./script.sh clean_logs
```
Moves all log files to an `archive` subdirectory inside the `logs/` folder.

### 9. Display Help
```bash
./script.sh help
```
Displays a summary of available commands and their usage.

## Log Files
All training logs are stored in the `logs/` directory with filenames formatted as:
```
<environment_type>_<influence_mode>_exp-<instance_number>-<timestamp>.log
```

## Job Tracking
- PIDs for running jobs are stored in `train_pids_<environment>_<influence>.txt`.
- Jobs are monitored and managed through these files.

## Notes
- The script ensures that log directories exist before execution.
- Uses `nohup` to run jobs in the background, preventing termination on session exit.
- Job details include timestamps and unique instance IDs for easy tracking.

## Example Workflow
1. Start training jobs:
   ```bash
   ./script.sh start multi-agent perfect-information
   ```
2. List active jobs:
   ```bash
   ./script.sh list multi-agent perfect-information
   ```
3. Query details of a job:
   ```bash
   ./script.sh query multi-agent perfect-information 12345
   ```
4. Kill a specific job:
   ```bash
   ./script.sh kill multi-agent perfect-information 12345
   ```
5. Stop all jobs for an experiment:
   ```bash
   ./script.sh stop multi-agent perfect-information
   ```
6. Archive logs:
   ```bash
   ./script.sh clean_logs
   ```

