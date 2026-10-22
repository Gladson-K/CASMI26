"""Search space definition for the retrieval + ranker NAS."""
import optuna

def define_search_space(trial: optuna.Trial) -> dict:
    return {
        'precursor_tolerance_ppm': trial.suggest_float('precursor_tolerance_ppm', 5.0, 20.0, step=2.5),
        'similarity_metric': trial.suggest_categorical('similarity_metric', ['cosine', 'modified_cosine']),
        'mz_tolerance': trial.suggest_float('mz_tolerance', 0.005, 0.05, step=0.005),
        'intensity_power': trial.suggest_float('intensity_power', 0.5, 2.0, step=0.25),
        'aggregation_method': trial.suggest_categorical('aggregation_method', ['max', 'mean', 'top3_mean']),
        'ranker_type': trial.suggest_categorical('ranker_type', ['gbm', 'rf', 'mlp']),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'hidden_layer_1': trial.suggest_int('hidden_layer_1', 32, 256, step=32),
        'hidden_layer_2': trial.suggest_int('hidden_layer_2', 16, 128, step=16),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'use_precursor_mass_feature': trial.suggest_categorical('use_precursor_mass_feature', [True, False]),
        'use_peak_count': trial.suggest_categorical('use_peak_count', [True, False]),
    }