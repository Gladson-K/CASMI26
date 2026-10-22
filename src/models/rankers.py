"""Ranker model factory."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_ranker(params):
    rt = params['ranker_type']
    if rt == 'gbm':
        return GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=42,
        )
    if rt == 'rf':
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42,
            n_jobs=-1,
        )
    if rt == 'mlp':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(params['hidden_layer_1'], params['hidden_layer_2']),
                alpha=params['alpha'],
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ))
        ])
    raise ValueError(f"Unknown ranker type: {rt}")