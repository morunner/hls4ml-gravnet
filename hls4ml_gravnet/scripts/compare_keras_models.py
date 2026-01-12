import pandas as pd
from keras.models import Model
from qgravnet.factory import QGravNetFactory
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from utils.data import load_processed
from utils.evaluation import load_run, response_rmse_log
from utils.files import RESULTS_PATH

TRAIN_DIRS = ['default', 'nn_8', 'nn_8_pruned']


def add_model_predictions_to_df(keras_model: Model, D: dict, df: pd.DataFrame, name: str) -> pd.DataFrame:
    test_energy_pred, test_pid_pred = keras_model.predict(D['X_hits_test'])

    test_response_rmse = response_rmse_log(D['y_energy_test'], test_energy_pred)
    test_auc = roc_auc_score(D['y_pid_test'], test_pid_pred)

    row = pd.Series(
        {
            'model': name,
            'auc': test_auc,
            'rmse': test_response_rmse,
        }
    )
    df = pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True)
    return df


def main():
    df = pd.DataFrame(columns=['model', 'auc', 'rmse'])
    for train_dir in TRAIN_DIRS:
        model_cfg, weights_path, history, datapath = load_run(RESULTS_PATH / train_dir)

        D = load_processed(datapath)

        trained_model = QGravNetFactory(**model_cfg).create_keras_model(128, 4)
        trained_model.load_weights(weights_path)
        trained_model.compile()

        df = add_model_predictions_to_df(trained_model, D, df, train_dir)

    # Print metrics
    print(tabulate(df.round(3), headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    main()
