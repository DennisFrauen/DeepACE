import plotting.plots as plotting
import joblib
from pathlib import Path
import os

if __name__ == "__main__":
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    save_dir = str(path.parent.absolute()) + "/results/df_results.pkl"
    df_results = joblib.load(save_dir)
    plotting.plot_results(df_results)