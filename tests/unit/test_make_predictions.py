import unittest
import pandas as pd
import numpy as np
import os

os.chdir('/Users/stevengeorge/Documents/Github/fpl-analysis')  # TODO Fix paths

from src.models.make_predictions import make_predictions


class TestMakePredictions(unittest.TestCase):
    def setUp(self):
        self.input_data = pd.read_parquet('tests/data/make_predictions_input_data.parquet')

    def test_same_predictions_on_multiple_runs(self):
        """
        Check that predictions are same on 5 repeated runs
        """
        prediction_outputs = []
        i = 0
        while i < 5:
            predictions_df = make_predictions(
                previous_gw=4,
                prediction_season_order=3,
                save_file=False,
                input_df=self.input_data
            )
            prediction_outputs.append(predictions_df['predictions'])
            i += 1

        for array in range(len(prediction_outputs) - 1):
            np.testing.assert_array_equal(
                prediction_outputs[array],
                prediction_outputs[array+1]
            )

    def test_same_predictions_on_shuffled_run(self):
        """
        Check that predictions are same when rows in DataFrame are shuffled
        """
        unshuffled_predictions_df = make_predictions(
            previous_gw=4,
            prediction_season_order=3,
            save_file=False,
            input_df=self.input_data
        )
        unshuffled_predictions = unshuffled_predictions_df['predictions']

        shuffled_input_data = self.input_data.sample(frac=1)
        shuffled_predictions_df = make_predictions(
            previous_gw=4,
            prediction_season_order=3,
            save_file=False,
            input_df=shuffled_input_data
        )
        shuffled_predictions = shuffled_predictions_df['predictions']

        np.testing.assert_array_equal(unshuffled_predictions, shuffled_predictions)

    def test_same_prediction_when_new_rows_added(self):
        """
        Check that prediction for a given player is the same even if additional rows are added to DataFrame
        """
        predictions_df = make_predictions(
            previous_gw=4,
            prediction_season_order=3,
            save_file=False,
            input_df=self.input_data
        )
        salah_pred_1 = predictions_df[
            (predictions_df['name'] == 'mohamed_salah')
        ]['predictions'].item()

        additional_rows = self.input_data[self.input_data['name'] != 'mohamed_salah'].sample(frac=0.01)
        input_data_xl = self.input_data.append(additional_rows)
        input_data_xl = input_data_xl.sample(frac=1)
        predictions_df_xl = make_predictions(
            previous_gw=4,
            prediction_season_order=3,
            save_file=False,
            input_df=input_data_xl
        )
        salah_pred_2 = predictions_df_xl[
            (predictions_df_xl['name'] == 'mohamed_salah')
        ]['predictions'].item()

        self.assertEqual(salah_pred_1, salah_pred_2)
