from utils import database as db
from utils import visualize as vs

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import gamma
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


def label_encoder(_data: pd.DataFrame) -> pd.DataFrame:
    maps = {
        "0": 0, "15": 1, "30": 2, "40": 3, "AD": 4,

        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,

        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9
    }

    data = _data.copy()
    data["p1_score"] = data["p1_score"].map(maps)
    data["p2_score"] = data["p2_score"].map(maps)
    data["p1_score"].drop_duplicates()

    data["p1_score_diff"] = data["p1_score"] - data["p2_score"]
    data["p2_score_diff"] = data["p2_score"] - data["p1_score"]

    return data

def normalize(_data: pd.DataFrame) -> pd.DataFrame:
    data_norm = _data.copy()

    # Run Distance

    rd = _data["p1_distance_run"]._append(_data["p2_distance_run"])
    fit_params = gamma.fit(rd)
    a, _, b = fit_params
    # nor = np.sqrt(a)*b
    nor = 2 * a * b

    data_norm["p1_distance_run"] = _data["p1_distance_run"] / nor
    data_norm["p2_distance_run"] = _data["p2_distance_run"] / nor

    # Rally Count

    rc = _data["rally_count"]

    try:
        fit_params = gamma.fit(rc)
        a, _, b = fit_params
    except:
        a, b = np.mean(rc), 1
    # nor = np.sqrt(a)*b
    nor = 2 * a * b
    data_norm["rally_count"] = _data["rally_count"] / nor

    # Speed

    sp = _data["speed_mph"]
    fit_params = norm.fit(sp)
    a, b = fit_params
    # nor = np.sqrt(a)*b
    nor = 2 * a
    data_norm["speed_mph"] = _data["speed_mph"].fillna(0) / nor

    data_norm['p1_ser_sp'] = data_norm['speed_mph'] * data_norm['p1_ser']
    data_norm['p2_ser_sp'] = data_norm['speed_mph'] * data_norm['p2_ser']

    return data_norm


class ModelUnit():
    def __init__(self, p1: str, p2: str, data: pd.DataFrame):
        import warnings
        # Ignore all warnings
        warnings.filterwarnings("ignore")

        self.p1 = p1
        self.p2 = p2
        self.data = db.select_match_norm(data, p1, p2)
        self.filtered = False
        self.filter_tar = None
        self.set_range = db.get_set_range(self.data)

        warnings.resetwarnings()

        self.predictions = None

    def preprocess(self):
        self.data = label_encoder(self.data)
        self.data = normalize(self.data)

    def prefilter(self, tars: list[str] = None):
        if self.filtered:
            pass

        if tars is None:
            tars = ["p1_distance_run", "p2_distance_run",
                    "rally_count",
                    'p1_game_win', 'p2_game_win',
                    'p1_unf_err', 'p2_unf_err',
                    "speed_mph"]

        for tar in tars:
            tar_ser = self.data[tar]
            tar_ser = db.expected_P(tar_ser, 0.5, 30)
            self.data[tar] = tar_ser.values

        self.filter_tar = tars
        self.filtered = True

        win = ["p1_win","p2_win"]
        for tar in win:
            tar_ser = self.data[tar]
            tar_ser = db.expected_P(tar_ser, 0.2, 30)
            self.data[tar] = tar_ser.values

    def show_components(self):
        plt.figure(figsize=(15, 6))

        plt.plot(self.data[self.filter_tar], label=self.filter_tar)
        vs.set_label(r"Normalized Torque Components", r"Duration $t$", r"Torque  $\tau_i$")

        plt.legend()

    def calc_torque(self, inertia: pd.DataFrame):
        I1 = inertia[self.p1]
        I2 = inertia[self.p2]

        w1, w2 = db.get_velocity(self.data['p1_win'], I1, I2)
        self.L1, self.L2 = I1 * w1, I2 * w2
        DL1, DL2 = self.L1.diff().fillna(0), self.L2.diff().fillna(0)
        Dt = DL1.index.diff().fillna(pd.to_timedelta("00:01:00"))
        # Dt = db.to_minute(Dt)
        self.Dt = Dt.to_series().apply(lambda x: x.total_seconds() / 60).values

        self.T1, self.T2 = DL1 / self.Dt, DL2 / self.Dt

    def show_calculated_torque(self):
        plt.figure(figsize=(15, 6))

        self.T1[5:].plot()
        vs.draw_range(self.set_range)
        vs.set_label("Torque over Sets", r"Duration $t$", r"Torque $N\cdot m$")

    def show_winning_rate(self):
        plt.figure(figsize=(15, 6))

        self.data['p1_win'].plot()
        vs.draw_range(self.set_range)
        vs.set_label(r"Current Winning Rate", r"Duration $t$", r"Winning Rate  $P_{k,t}$")

    def train(self, inputs:list[str]):
        self.model = LinearRegression()
        self.inputs = inputs

        x_torque = self.data[inputs].replace([np.inf, -np.inf, np.nan], 0).values
        y_torque = self.T1.replace([np.inf, -np.inf, np.nan], 0).values
        # Fit the model to the data
        self.model.fit(x_torque, y_torque)

        # Print the coefficients (slope and intercept)
        print("Slope (Coefficient):", self.model.coef_)
        print("Intercept:", self.model.intercept_)

        self.result = pd.DataFrame([inputs, self.model.coef_]).T.set_index(0)
        return self.result

    def show_params(self, sort=False):

        if sort:
            result = self.result.sort_values(by=1, ascending=False)
        else:
            result = self.result

        plt.figure(figsize=(15, 6))
        plt.bar(result.index, result[1])

        vs.set_label(r"Estimated Controlling Parameters", r"Components $\tau_i$", r"Parameters $\beta_i$")

    def predict(self):

        x_torque = self.data[self.inputs].replace([np.inf, -np.inf, np.nan], [-10,10,0]).values
        # Make predictions using the model
        self.predictions = self.model.predict(x_torque)

        return self.predictions

    def construct_momentum(self):
        if self.predictions is None:
            self.predict()

        self.L1c = np.cumsum(self.predictions * self.Dt)

        compare = pd.DataFrame(self.L1)
        compare["constructed momentum"] = self.L1c

        plt.figure(figsize=(15, 6))
        plt.plot(compare["p1_win"], label="calculated mumentum")
        plt.plot(compare["constructed momentum"], label="constructed mumentum")

        vs.draw_range(self.set_range)
        plt.legend()
        vs.set_label(r"Momentum Construction", r"Duration $t$", r"Momentum  $L_{k,t}$")