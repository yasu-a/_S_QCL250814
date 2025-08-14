import re
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(slots=True)
class IterationRecord:
    n_iter: int
    elapsed_time: float
    cost: float

    def to_json(self):
        return {
            "n_iter": self.n_iter,
            "elapsed_time": self.elapsed_time,
            "cost": self.cost,
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            n_iter=body["n_iter"],
            elapsed_time=body["elapsed_time"],
            cost=body["cost"],
        )


@dataclass(slots=True)
class GlobalRecord:
    nqubit: int
    c_depth: int
    time_step: float

    obs_coeff: float

    func_type: Literal["gauss10", "gauss5", "gauss3", "tri"]
    method: Literal["conv", "pow_3", "nonint"]

    opt_method: Literal["Nelder-Mead", "BFGS", "SLSQP"]
    opt_options: str

    seed_train: int
    seed_system: int
    seed_time_evol: int
    seed_theta_init: int

    x_train: np.ndarray
    y_train: np.ndarray

    x_test: np.ndarray

    theta_init: np.ndarray
    y_pred_init: np.ndarray
    cost_init: float

    theta_opt: np.ndarray
    y_pred_opt: np.ndarray
    cost_opt: float

    iteration_records: list[IterationRecord]

    def to_json(self):
        return {
            "nqubit": self.nqubit,
            "c_depth": self.c_depth,
            "time_step": self.time_step,

            "obs_coeff": self.obs_coeff,

            "func_type": self.func_type,
            "method": self.method,

            "opt_method": self.opt_method,
            "opt_options": self.opt_options,

            "seed_train": self.seed_train,
            "seed_system": self.seed_system,
            "seed_time_evol": self.seed_time_evol,
            "seed_theta_init": self.seed_theta_init,

            "x_train": self.x_train.tolist(),
            "y_train": self.y_train.tolist(),

            "x_test": self.x_test.tolist(),

            "theta_init": self.theta_init.tolist(),
            "y_pred_init": self.y_pred_init.tolist(),
            "cost_init": self.cost_init,

            "theta_opt": self.theta_opt.tolist(),
            "y_pred_opt": self.y_pred_opt.tolist(),
            "cost_opt": self.cost_opt,

            "iteration_records": [x.to_json() for x in self.iteration_records],
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            nqubit=body["nqubit"],
            c_depth=body["c_depth"],
            time_step=body["time_step"],

            obs_coeff=body["obs_coeff"],

            func_type=body["func_type"],
            method=body["method"],

            opt_method=body["opt_method"],
            opt_options=body["opt_options"],

            seed_train=body["seed_train"],
            seed_system=body["seed_system"],
            seed_time_evol=body["seed_time_evol"],
            seed_theta_init=body["seed_theta_init"],

            x_train=np.array(body["x_train"]),
            y_train=np.array(body["y_train"]),

            x_test=np.array(body["x_test"]),

            theta_init=np.array(body["theta_init"]),
            y_pred_init=np.array(body["y_pred_init"]),
            cost_init=body["cost_init"],

            theta_opt=np.array(body["theta_opt"]),
            y_pred_opt=np.array(body["y_pred_opt"]),
            cost_opt=body["cost_opt"],

            iteration_records=[IterationRecord.from_json(x) for x in body["iteration_records"]],
        )

    @property
    def opt_maxiter(self):
        return int(re.search(r"\bmaxiter=(\d+)\b", self.opt_options).group(1))
