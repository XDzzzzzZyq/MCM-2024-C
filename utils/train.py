from utils import database as db
from utils import visualize as vs

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
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

        self.p1_gwin_time = self.data[self.data['p1_game_win']].index
        self.p2_gwin_time = self.data[self.data['p2_game_win']].index

        warnings.resetwarnings()

        self.predictions1 = self.predictions2 = None

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

        vs.set_caption()

    def calc_torque(self, inertia: pd.DataFrame):
        self.I1 = inertia[self.p1]
        self.I2 = inertia[self.p2]

        w1, w2 = db.get_velocity(self.data['p1_win'], self.I1, self.I2)
        self.L1, self.L2 = self.I1 * w1, self.I2 * w2
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
        vs.set_label(r"Current Winning Rate", r"Duration $t$", r"Winning Rate  $P_{k}(t)$")

    def train(self, inputs: dict[str, list[str]], ratio:float=1.0):
        self.model1 = LinearRegression()
        self.model2 = LinearRegression()
        self.inputs = inputs

        inp1 = self.inputs["p1"]
        inp2 = self.inputs["p2"]

        r = int(ratio*len(self.data))

        x_torque1 = self.data[inp1].replace([np.inf, -np.inf, np.nan], 0).values
        y_torque1 = self.T1.shift(0).replace([np.inf, -np.inf, np.nan], 0).values
        # Fit the model to the data
        self.model1.fit(x_torque1[:r], y_torque1[:r])
        self.result1 = pd.DataFrame([inp1, self.model1.coef_]).T.set_index(0)


        x_torque2 = self.data[inp2].replace([np.inf, -np.inf, np.nan], 0).values
        y_torque2 = self.T2.shift(0).replace([np.inf, -np.inf, np.nan], 0).values
        # Fit the model to the data
        self.model2.fit(x_torque2[:r], y_torque2[:r])
        self.result2 = pd.DataFrame([inp2, self.model2.coef_]).T.set_index(0)

        return self.result1, self.result2

    def show_params(self, sort=False, pl:int = 1):

        if sort:
            result = (self.result1 if pl is 1 else self.result2).sort_values(by=1, ascending=False)
        else:
            result = (self.result1 if pl is 1 else self.result2)

        plt.figure(figsize=(15, 6))
        plt.bar(result.index, result[1])

        vs.set_label(r"Estimated Controlling Parameters", r"Components $\tau_i$", r"Parameters $\beta_i$")

    def predict(self):

        inp1 = self.inputs["p1"] + self.inputs["gen"]
        inp2 = self.inputs["p2"] + self.inputs["gen"]

        x_torque1 = self.data[inp1].replace([np.inf, -np.inf, np.nan], [-10,10,0]).values
        x_torque2 = self.data[inp2].replace([np.inf, -np.inf, np.nan], [-10,10,0]).values
        # Make predictions using the model
        self.predictions1 = self.model1.predict(x_torque1)
        self.predictions2 = self.model2.predict(x_torque2)

        return x_torque1, x_torque2, self.predictions1, self.predictions2

    def predict_ex(self, x_torque1, x_torque2):
        predictions1 = self.model1.predict(x_torque1)
        predictions2 = self.model2.predict(x_torque2)

        return predictions1, predictions2

    def construct_momentum(self):
        if self.predictions1 is None:
            self.predict()

        self.L1c = np.cumsum(self.predictions1 * self.Dt)
        self.L2c = np.cumsum(self.predictions2 * self.Dt)

        self.compare = pd.DataFrame(self.L1)
        self.compare["constructed 1"] = self.L1c
        self.compare["constructed 2"] = self.L2c
        self.compare["sum"] = self.L2c + self.L1c

    @classmethod
    def _cons_ener(cls, L1c, L2c, I1, I2, Off1 = 0, Off2 = 0):
        E1c = (L1c + Off1) ** 2 / I1 / 2
        E2c = (L2c + Off2) ** 2 / I2 / 2
        return E1c, E2c
    def construct_energy(self):
        E1c, E2c = ModelUnit._cons_ener(self.L1c, self.L2c, self.I1, self.I2)

        self.compare['E1c'] = E1c
        self.compare['E2c'] = E2c
        self.compare['E/E'] = E1c / E2c

    stab = [[0, 0], [0.5, 0.5], [0.8, 0.2], [0.2, 0.8]]
    def construct_energy_stab(self):

        for idx, stb in enumerate(ModelUnit.stab):
            E1c, E2c = ModelUnit._cons_ener(self.L1c, self.L2c, self.I1, self.I2, stb[0], stb[1])

            self.compare[f'E/E {idx}'] = E1c / E2c


    def compare_performance(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.compare['E/E'][self.compare['E/E']<5], label=r"performance coefficient $p$")
        plt.axhline(y=1, color='red', linestyle='--', label=r'$p=1$')
        vs.draw_range(self.set_range)
        vs.set_caption()
        vs.set_label(r"Performance Coefficient $p$", r"Duration $t$", r"$p_{k}(t)$")
        vs.set_xaxis()
        vs.set_caption(True, False, True)

    def compare_performance_stab(self):
        plt.figure(figsize=(15, 6))
        stab = [[0,0], [0.5,0.5], [1,0], [0,1]]
        for idx, stb in enumerate(ModelUnit.stab):
            plt.plot(self.compare[f'E/E {idx}'][self.compare[f'E/E {idx}']<5], label=fr"$L_1(0) = {stb[0]}, L_2(0) = {stb[1]}$")
        plt.axhline(y=1, color='red', linestyle='--', label=r'$p=1$')
        vs.draw_range(self.set_range)
        vs.set_caption()
        vs.set_label(r"Performance Coefficient $p$", r"Duration $t$", r"$p_{k}(t)$")
        vs.set_xaxis()
        vs.set_caption(True, False, True)

    def show_constructed(self, p:int):
        if p not in [1,2]:
            pass

        plt.figure(figsize=(15, 6))
        plt.plot(self.data[f"p{p}_win"], label="calculated mumentum")
        plt.plot(self.compare[f"constructed {p}"], label="constructed mumentum")

        vs.draw_range(self.set_range)
        vs.set_caption()
        vs.set_label(r"Momentum Construction", r"Duration $t$", r"Momentum  $L_{k}(t)$")

    def compare_constructed(self):
        plt.figure(figsize=(15, 6))
        # plt.plot(compare["p1_win"], label="calculated mumentum")
        plt.plot(self.compare["constructed 2"], label=self.p2)
        plt.plot(self.compare["constructed 1"], label=self.p1)
        plt.plot(self.compare["sum"], label="Total Momentum")
        #plt.plot(self.data["p1_unf_err"], label="Unforced Error")
        #plt.plot(self.data["speed_mph"], label="Serve Speed")

        plt.scatter(x=self.p2_gwin_time, y=self.compare["constructed 2"][self.p2_gwin_time])
        plt.scatter(x=self.p1_gwin_time, y=self.compare["constructed 1"][self.p1_gwin_time])

        vs.draw_range(self.set_range)
        vs.set_caption()
        vs.set_label(r"Momentum Comparison", r"Duration $t$", r"Momentum  $L_{k}(t)$")

    def compare_migrate(self, mig:np.ndarray):
        mig = np.cumsum(mig * self.Dt)
        self.compare["mig 2"] = mig

        plt.figure(figsize=(15, 6))
        # plt.plot(compare["p1_win"], label="calculated mumentum")
        plt.plot(self.compare["mig 2"], label=self.p1+" (Mig)")
        plt.plot(self.compare["constructed 1"], label=self.p1)
        # plt.plot(self.data["p1_unf_err"], label="Unforced Error")
        # plt.plot(self.data["speed_mph"], label="Serve Speed")

        vs.draw_range(self.set_range)
        vs.set_caption()
        vs.set_label(r"Model Migration", r"Duration $t$", r"Momentum  $L_{k}(t)$")
        vs.set_xaxis()

    def compare_indicator(self, name:str, label:str):
        plt.figure(figsize=(15, 6))
        # plt.plot(compare["p1_win"], label="calculated mumentum")
        #plt.plot(self.compare["constructed 2"], label=self.p2)

        #plt.plot(self.compare["sum"], label="Total Momentum")
        # plt.plot(self.data["speed_mph"], label="Serve Speed")
        plt.plot(self.compare["constructed 1"], label=self.p1)
        #plt.scatter(x=self.p2_gwin_time, y=self.compare["constructed 2"][self.p2_gwin_time])
        plt.scatter(x=self.p1_gwin_time, y=self.compare["constructed 1"][self.p1_gwin_time])
        plt.plot(self.data["p1_unf_err"], label="Unforced Error")
        plt.plot(self.data[name], label=label)
        vs.draw_range(self.set_range)
        vs.set_caption()
        vs.set_caption(True, False, True)
        vs.set_label(r"Indicator Analysis", r"Duration $t$", r"Momentum (components)  $L_{k}(t)$")



    def corr(self):

        corr = self.compare[["constructed 1", "constructed 2"]]
        corr["winning rate 1"] = self.data[f"p1_win"].shift(1).fillna(0)
        corr["winning rate 2"] = self.data[f"p2_win"].shift(1).fillna(0)

        corr_matrix = corr.corr()
        plt.figure(figsize=(7, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix', fontsize = 20)

    def corr2(self):

        corr = self.compare[["constructed 1", "constructed 2"]]
        corr["winning rate 1"] = self.data[f"p1_win"]
        corr["winning rate 2"] = self.data[f"p2_win"]
        corr["performance"] = self.compare['E/E']


        corr_matrix = corr.corr()
        plt.figure(figsize=(7, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')

    def summary(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(15, 12))

        ax1.plot(self.compare["constructed 2"], label=self.p2)
        ax1.plot(self.compare["constructed 1"], label=self.p1)
        ax1.plot(self.compare["sum"], label="Total Momentum")
        # plt.plot(self.data["p1_unf_err"], label="Unforced Error")
        # plt.plot(self.data["speed_mph"], label="Serve Speed")

        ax1.scatter(x=self.p2_gwin_time, y=self.compare["constructed 2"][self.p2_gwin_time])
        ax1.scatter(x=self.p1_gwin_time, y=self.compare["constructed 1"][self.p1_gwin_time])

        vs.draw_range(self.set_range, ax1)
        vs.set_caption(True, True, True, ax1)
        vs.set_label(r"Momentum Comparison", r"Duration $t$", r"Momentum  $L_{k}(t)$", ax1)
        ax1.set_xlabel(None)

        ax2.plot(self.compare['E/E'][self.compare['E/E'] < 5], label=r"performance coefficient $p$")
        ax2.axhline(y=1, color='red', linestyle='--', label=r'$p=1$')
        vs.draw_range(self.set_range, ax2)
        vs.set_caption(True, False, True, ax2)
        vs.set_label(r"Performance Coefficient $p$", r"Duration $t$", r"$p_{k}(t)$", ax2)
        vs.set_xaxis()

        #fig.suptitle('Momentum Comparison', fontsize=20)
