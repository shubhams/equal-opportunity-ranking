# equal-opportunity-ranking

Getting started
===============
To get started, you can install the package using pip:

## Virtual Environment (Recommended)
Create a virtual environment using `uv` (optional but recommended):
```bash
uv venv eopp-ranking
source eopp-ranking/bin/activate
```
To deactivate the virtual environment, simply run:
```bash
deactivate
```
The repo has a `requirements.txt` that can be used to install the necessary dependencies. You can install them using pip:
```bash
uv pip install -r requirements.txt
```

## Running the Code
### Simulate Cut Points 
You can run the code using the command line. For example, to run the `exp_cutpts.py` script, you can use the following command:
```bash
python exp_cutpts.py --alpha 0.0 --m 100 --k 6 --just_plot
```
`m` is the number of samples, `k` is the number of documents to rank, and `alpha` is the weight on unfairness in the objective. The `--just_plot` flag can be used to skip the computation and just plot the results from previously computed CSV files.

Another file, `ranker.py`, contains code for computing the utility and unfairness of different rankings.

### Analytical Solution
Fixing group A and B documents to a constant, we can solve for the optimal cut points analytically. This is implemented in `analytical_solution.py`. You can run it using:
```bash
python analytical_solution.py --k 6 --r 3 
```