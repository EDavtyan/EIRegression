import optuna
from optuna.pruners import BasePruner
import optuna.integration
import optuna.trial
from optuna.study import StudyDirection
from optuna.trial import TrialState
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
optuna.logging.set_verbosity(optuna.logging.DEBUG)


class ModelOptimizer:
    def __init__(
            self,
            n_trials: int = 500,
            timeout: int = 600,
            metric: str = 'r2',
            early_stopping_rounds: Optional[int] = 20,
            random_state: int = 42
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.study = None
        self.best_pipeline = None
        self.cv_results_ = None

    def _get_metric_func(self):
        if self.metric == 'r2':
            return r2_score
        elif self.metric == 'neg_mse':
            return lambda y, y_pred: -mean_squared_error(y, y_pred)
        elif self.metric == 'neg_mae':
            return lambda y, y_pred: -mean_absolute_error(y, y_pred)
        else:
            return self.metric

    def _create_objective(self, pipeline, param_distributions, X_train, y_train, cv_strategy, scoring):
        def objective(trial):
            params = {}
            for param_name, param_info in param_distributions.items():
                if isinstance(param_info, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_info)
                elif isinstance(param_info, tuple):
                    if isinstance(param_info[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_info[0], param_info[1])
                    elif param_name.endswith(('learning_rate', 'alpha', 'lambda')):
                        params[param_name] = trial.suggest_float(param_name, param_info[0], param_info[1], log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_info[0], param_info[1])

            pipeline.set_params(**params)

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring=scoring if scoring is not None else self._get_metric_func(),
                cv=cv_strategy,
                n_jobs=-1
            )

            trial.set_user_attr('cv_scores_mean', scores.mean())
            trial.set_user_attr('cv_scores_std', scores.std())

            return scores.mean()

        return objective

    class EarlyStoppingCallback:
        def __init__(self, early_stopping_rounds: int):
            self.early_stopping_rounds = early_stopping_rounds
            self.best_value = None
            self.best_trial = None
            self.stagnant_trials = 0

        def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.state != TrialState.COMPLETE:
                return

            current_value = trial.value

            if self.best_value is None or current_value > self.best_value:
                self.best_value = current_value
                self.best_trial = trial.number
                self.stagnant_trials = 0
            else:
                self.stagnant_trials += 1

            if self.stagnant_trials >= self.early_stopping_rounds:
                study.stop()

    def _get_random_param_value(self, param_name: str, param_info: Any) -> Any:
        """
        Helper method to get random parameter value based on the parameter type

        Args:
            param_name: Name of the parameter
            param_info: Parameter distribution information

        Returns:
            Randomly sampled parameter value
        """
        if isinstance(param_info, list):
            return np.random.choice(param_info)
        elif isinstance(param_info, tuple):
            if isinstance(param_info[0], int):
                return np.random.randint(param_info[0], param_info[1])
            elif isinstance(param_info[0], float):
                if any(substr in param_name.lower() for substr in ('learning_rate', 'alpha', 'lambda')):
                    log_min, log_max = np.log(param_info[0]), np.log(param_info[1])
                    return np.exp(np.random.uniform(log_min, log_max))
                return np.random.uniform(param_info[0], param_info[1])
        elif isinstance(param_info, dict):
            if 'type' in param_info:
                if param_info['type'] == 'int':
                    # For integer parameters, include the high value in the range
                    return np.random.randint(param_info['low'], param_info['high'] + 1)
                elif param_info['type'] == 'float':
                    return np.random.uniform(param_info['low'], param_info['high'])
                elif param_info['type'] == 'categorical':
                    return np.random.choice(param_info['choices'])
            elif 'distribution' in param_info:
                dist_type = param_info['distribution']
                if dist_type == 'uniform':
                    return np.random.uniform(param_info['low'], param_info['high'])
                elif dist_type == 'loguniform':
                    log_min, log_max = np.log(param_info['low']), np.log(param_info['high'])
                    return np.exp(np.random.uniform(log_min, log_max))
                elif dist_type == 'int_uniform':
                    return np.random.randint(param_info['low'], param_info['high'] + 1)
                elif dist_type == 'choice':
                    return np.random.choice(param_info['choices'])
            elif all(k in param_info for k in ['low', 'high']):
                if isinstance(param_info['low'], int) and isinstance(param_info['high'], int):
                    return np.random.randint(param_info['low'], param_info['high'] + 1)
                return np.random.uniform(param_info['low'], param_info['high'])
            elif 'choices' in param_info:
                return np.random.choice(param_info['choices'])

        raise ValueError(f"Unsupported parameter distribution type: {type(param_info)} "
                         f"with content: {param_info}")


    def optimize(
            self,
            pipeline,
            param_distributions: Dict[str, Any],
            X_train,
            y_train,
            cv: int = 5,
            scoring: Optional[str] = None,
            lower_search_bound: int = 5,
            **kwargs
    ):
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            pipeline: Pipeline including preprocessing and the model
            param_distributions: Dictionary of parameter distributions for optimization
            X_train: Training data features
            y_train: Training data target
            cv: Number of folds for cross-validation
            scoring: Scoring metric to use (if None, uses the metric specified in __init__)
            lower_search_bound: Minimum number of samples required for optimization
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            The pipeline with the best found parameters or None if training data is empty
        """
        # Check if training data is empty
        if X_train.shape[0] == 0 or len(y_train) == 0:
            print("Warning: Empty training data detected, returning None")
            return None

        # Handle case where data is below lower search bound
        if len(y_train) <= lower_search_bound:
            try:
                random_params = {
                    k: self._get_random_param_value(k, v)
                    for k, v in param_distributions.items()
                }
                pipeline.set_params(**random_params)
                pipeline.fit(X_train, y_train)
                return pipeline
            except Exception as e:
                print(f"Error sampling random parameters or fitting model: {str(e)}. "
                      f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
                return None

        # Regular optimization process for sufficient data
        n_splits = min(cv, len(y_train))
        cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        objective = self._create_objective(
            pipeline, param_distributions, X_train, y_train, cv_strategy, scoring
        )

        callbacks = []
        if self.early_stopping_rounds is not None:
            early_stopping = self.EarlyStoppingCallback(self.early_stopping_rounds)
            callbacks.append(early_stopping)

        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        try:
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                callbacks=callbacks,
                n_jobs=-1
            )

            self.cv_results_ = pd.DataFrame({
                'trial': range(len(self.study.trials)),
                'value': [t.value for t in self.study.trials],
                'cv_scores_mean': [t.user_attrs.get('cv_scores_mean') for t in self.study.trials],
                'cv_scores_std': [t.user_attrs.get('cv_scores_std') for t in self.study.trials],
                **{k: [t.params.get(k) for t in self.study.trials] for k in param_distributions.keys()}
            })

            best_params = self.study.best_params
            pipeline.set_params(**best_params)
            pipeline.fit(X_train, y_train)
            self.best_pipeline = pipeline

            return pipeline

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return None