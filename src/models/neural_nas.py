"""Lightweight MLP architecture search on precomputed features."""
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def neural_nas_objective(trial, X, y, cv=3):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden = tuple(
        trial.suggest_int(f'h{i}', 16, 256, step=16) for i in range(n_layers)
    )
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch = trial.suggest_categorical('batch_size', [32, 64, 128])

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=activation,
            alpha=alpha,
            learning_rate_init=lr,
            batch_size=batch,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
    return float(scores.mean())

def run_neural_nas(X, y, n_trials=30, timeout=3600 * 4, seed=42):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name='casmi26_neural_nas',
    )
    study.optimize(
        lambda t: neural_nas_objective(t, X, y),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
    )
    return study