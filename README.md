How to load data:

# 1) Create a clean environment
python -m venv eotdl-env
source eotdl-env/bin/activate    # on Windows: eotdl-env\Scripts\activate

# 2) Install EOTDL
pip install eotdl

# 3) Log in
eotdl auth login

# 4) Check the dataset name
eotdl datasets list --name ClearSAR

# 5) Download it locally
eotdl datasets get ClearSAR --path ./data --assets
