from autogluon.tabular import TabularPredictor
import pandas as pd

train_data = pd.read_csv("input_data.csv")
predictor = TabularPredictor(label="target", problem_type="regression")    .fit(train_data=train_data, time_limit=30000)
leaderboard = predictor.leaderboard(silent=True)
leaderboard.to_csv("leaderboard.csv", index=False)
model = predictor