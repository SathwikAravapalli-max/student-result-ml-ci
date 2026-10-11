
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 85,
            "internal_marks": 75,
            "assignment_marks": 80,
            "previous_score": 78
        }])

        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_performance_student(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)

    def test_low_performance_student(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 55,
            "internal_marks": 30,
            "assignment_marks": 40,
            "previous_score": 35
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
