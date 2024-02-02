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


def get_velocity_2(P: pd.Series, I1: float, I2: float) -> [pd.Series, pd.Series]:
    I1I2 = I1*I2
    I1I2P = I1I2*P
    SQRT = np.sqrt(I1I2P-I1I2P*P)
    dom = -I1I2+I1**2*P+I1I2P

    w1 = (I1*P+SQRT)/dom
    w2 = 1-I1**2*P/dom - I1*SQRT/dom
    w2 /= I2

    return w1, w2


def _expected_P(P: pd.Series, l:float = 0.8) -> float:
    w = len(P)
    ker = l*np.power(1-l, np.arange(0, w))
    ker[-1] += (1-l)**(w)
    #print(ker)
    #print(sum(ker))
    return sum(P*ker)


def expected_P(P: pd.Series, l:float = 0.8, w:int = 15) -> pd.Series:
    return pd.Series([_expected_P(P[max(0, i-w+1):i+1], l) for i in range(len(P))])


if __name__ == '__main__':
    print(_expected_P([2,1,1,1,1,1,1], 0.8))

    print(len(expected_P([1,2,3,4,5,6,7,8,9,10], 0.8, 5)))
    expected_P([1,2,3,4,5,6,7,8,9], 0.2, 5)

