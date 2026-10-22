"""Run full inference: load best params from NAS, generate submission."""
import yaml, json, pickle
import pandas as pd
from src.data.loader import load_competition
from src.data.preprocess import process_dataframe, make_spectrum
from src.submission.generate import generate_submission

def main():
    cfg = yaml.safe_load(open('config.yaml'))
    train, test = load_competition(cfg)
    train = process_dataframe(train, cfg)
    test = process_dataframe(test, cfg)

    # Load best params
    with open(f"{cfg['paths']['output_dir']}/best_params.json") as f:
        best_params = json.load(f)

    # Load library
    with open(f"{cfg['paths']['cache_dir']}/library.pkl", 'rb') as f:
        entries = pickle.load(f)

    lib_specs, lib_smiles = [], []
    for e in entries[:100000]:
        s = make_spectrum([e['mass']], [1.0], e['mass'])
        lib_specs.append(s)
        lib_smiles.append(e['smiles'])

    sub = generate_submission(train, test, lib_specs, lib_smiles, best_params,
                              top_n=cfg['submission']['top_n'])
    sub.to_csv(f"{cfg['paths']['output_dir']}/submission.csv", index=False)
    print(f"Submission written with {len(sub)} rows")

if __name__ == '__main__':
    main()