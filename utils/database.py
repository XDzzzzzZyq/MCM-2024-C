import pandas as pd
import numpy as np

def select_match(data: pd.DataFrame, player1: str, player2: str) -> [pd.DataFrame, pd.DataFrame]:
    eg1 = data[(data["player1"] == player1) & (data["player2"] == player2)].set_index("point_no")

    eg1["p1_score"][eg1["p1_score"] == 'AD'] = 60
    eg1["p2_score"][eg1["p2_score"] == 'AD'] = 60

    eg1['elapsed_time'] = pd.to_datetime(eg1['elapsed_time'], format='mixed')

    win_series = eg1['p1_score'].astype(int) != eg1["p1_score"].astype(int).shift(1).fillna(0)

    return eg1, win_series


def get_velocity(P: pd.Series, I1: float, I2: float) -> [pd.Series, pd.Series]:
    I1I2 = I1*I2
    I1I2P = I1I2*P
    SQRT = np.sqrt(I1I2P-I1I2P*P)
    dom = -I1I2+I1**2*P+I1I2P

    w1 = (I1*P-SQRT)/dom
    w2 = 1-I1**2*P/dom + I1*SQRT/dom
    w2 /= I2

    return w1, w2
