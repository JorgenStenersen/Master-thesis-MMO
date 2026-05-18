# master-thesis
For developing and solving the NMMWPBP

## Requirements

- Python 3.12.x
- Gurobi 12.0.x (must be installed separately)


## Setup

### macOS / Linux
<pre>
python -m venv thesis-env
source thesis-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
</pre>

### Windows
<pre>
python -m venv thesis-env
thesis-env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
</pre>

## Run
<pre>
python -m scripts.run_main
</pre>

## Project structure
<pre>
master-thesis/
│
├── scenred_backred                  # Data and scenario reduction
│
├── results/                # Generated results (not tracked by git)
│
├── scripts/
│   ├── run_main.py         # Entry point for running the model
|   └── ph_bundle_worker.py # Solves the ph bundles and returns results
|   └── ph_slurm_coordinator.py   # runs full ph algorithm with local parallelization
|   └── run_main_slurm.sh   # Batch script for SGE
│
├── src/                    # Core model code
│   ├── model.py
│   ├── tree.py
│   ├── read.py
│   └── utils.py
|   └── model_container.py
|   └── solvers /
|       ├── extensive_form.py
|       └── progressive_hedging.py
│
├── experiments/            # Experiment logic and test scripts
|
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore

</pre>

