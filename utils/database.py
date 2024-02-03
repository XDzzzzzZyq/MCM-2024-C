import pandas as pd
import numpy as np

def select_match(data: pd.DataFrame, player1: str, player2: str) -> [pd.DataFrame, pd.DataFrame]:
    eg1 = data[(data["player1"] == player1) & (data["player2"] == player2)].set_index("point_no")

    eg1["p1_score"][eg1["p1_score"] == 'AD'] = 60
    eg1["p2_score"][eg1["p2_score"] == 'AD'] = 60

    eg1['elapsed_time'] = pd.to_datetime(eg1['elapsed_time'], format='mixed')

    win_series = eg1['p1_score'].astype(int) != eg1["p1_score"].astype(int).shift(1).fillna(0)

    return eg1, win_series


def select_match_norm(data: pd.DataFrame, player1: str, player2: str) -> pd.DataFrame:
    
    eg1 = data[(data["player1"] == player1) & (data["player2"] == player2)]
    eg1['elapsed_time'] = pd.to_datetime(eg1['elapsed_time'], format='%H:%M:%S')
    
    eg1['p1_win'] = eg1['point_victor'] == 1
    eg1['p2_win'] = eg1['point_victor'] == 2
    eg1['p2_game_win'] = eg1['game_victor'] == 2
    eg1['p1_game_win'] = eg1['game_victor'] == 1

    eg1['speed_mph'] = eg1['speed_mph'].fillna(0)
    eg1['p1_ser'] = eg1['server'] == 1
    eg1['p2_ser'] = eg1['server'] == 2
    eg1['p1_ser_sp'] = eg1['speed_mph'] * eg1['p1_ser']
    eg1['p2_ser_sp'] = eg1['speed_mph'] * eg1['p2_ser']

    eg1['rally_count'] -= 2*eg1['rally_count']*eg1['p2_win']

    return eg1.sort_values(by='elapsed_time').set_index('elapsed_time')


def get_set_range(match: pd.DataFrame) -> list[pd.Timestamp]:

    match = match.copy()
    match["new set"] = match["set_no"] != match["set_no"].shift(1)
    new_set = []

    for _index, _row in match.iterrows():
        if _row["new set"]:
            new_set += [_index]
    new_set.append(max(match.index))

    return new_set


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
    w2 = (1-w1)/I2

    return w1, w2


def _expected_P(P: pd.Series, l:float = 0.2) -> float:
    w = len(P)
    ker = l*np.power(1-l, np.arange(0, w))
    ker[-1] += (1-l)**(w)
    #print(ker)
    #print(sum(ker))
    return sum(P*ker[::-1])


def expected_P(P: pd.Series, l:float = 0.2, w:int = 15) -> pd.Series:
    return pd.Series([_expected_P(P[max(0, i-w+1):i+1], l) for i in range(len(P))])


def to_minute(time: pd.DatetimeIndex) -> float:
    return time.hour*60 + time.minute + time.second/60


if __name__ == '__main__':
    print(_expected_P([2,1,1,1,1,1,1], 0.8))

    print(len(expected_P([1,2,3,4,5,6,7,8,9,10], 0.8, 5)))
    expected_P([1,2,3,4,5,6,7,8,9], 0.2, 5)

    dictionary = pd.read_csv("../data_dictionary.csv")
    data = pd.read_csv("../Wimbledon_featured_matches.csv")
    data = select_match_norm(data, "Carlos Alcaraz", "Nicolas Jarry")
    exp = expected_P(data["p1_distance_run"])
    print(exp.head())

    set_range = get_set_range(data)
    print(set_range)