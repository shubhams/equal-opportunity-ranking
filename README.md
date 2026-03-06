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
You can run the code using the command line. For example, to run the `moon_exp_cutpts.py` script, you can use the following command:
```bash
python moon_exp_cutpts.py --alpha 0.0 --m 100 --k 6 --just_plot
```