import re
from dataclasses import dataclass

import numpy as np

from model.value import MethodType, FuncType, OptMethodType


@dataclass(slots=True)
class IterationRecord:
    n_iter: int
    elapsed_time: float
    train_loss: float
    val_loss: float
    best_val_loss: float
    patience_counter: int

    def to_json(self):
        return {
            "n_iter": self.n_iter,
            "elapsed_time": self.elapsed_time,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
        }

    @classmethod
    def from_json(cls, body):
        return cls(
            n_iter=body["n_iter"],
            elapsed_time=body["elapsed_time"],
            train_loss=body["train_loss"],
            val_loss=body["val_loss"],
            best_val_loss=body["best_val_loss"],
            patience_counter=body["patience_counter"],
        )


@dataclass(slots=True)
class GlobalRecord:
    nqubit: int
    c_depth: int
    time_step: float

    obs_coeff: float

    func_type: FuncType
    method: MethodType

    opt_method: OptMethodType
    opt_options: str

    seed_train: int
    seed_system: int
    seed_time_evol: int
    seed_theta_init: int

    x_train: np.ndarray
    y_train: np.ndarray
    y_pred_train_init: np.ndarray
    y_pred_train_opt: np.ndarray    

    x_test: np.ndarray
    y_test: np.ndarray
    y_pred_test_init: np.ndarray
    y_pred_test_opt: np.ndarray

    theta_init: np.ndarray
    x_plot_init: np.ndarray
    y_plot_init: np.ndarray
    train_loss_init: float
    test_loss_init: float

    theta_opt: np.ndarray
    x_plot_opt: np.ndarray
    y_plot_opt: np.ndarray
    train_loss_opt: float
    test_loss_opt: float

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
            "y_pred_train_init": self.y_pred_train_init.tolist(),
            "y_pred_train_opt": self.y_pred_train_opt.tolist(),

            "x_test": self.x_test.tolist(),
            "y_test": self.y_test.tolist(),
            "y_pred_test_init": self.y_pred_test_init.tolist(),
            "y_pred_test_opt": self.y_pred_test_opt.tolist(),

            "theta_init": self.theta_init.tolist(),
            "x_plot_init": self.x_plot_init.tolist(),
            "y_plot_init": self.y_plot_init.tolist(),
            "train_loss_init": self.train_loss_init,
            "test_loss_init": self.test_loss_init,

            "theta_opt": self.theta_opt.tolist(),
            "x_plot_opt": self.x_plot_opt.tolist(),
            "y_plot_opt": self.y_plot_opt.tolist(),
            "train_loss_opt": self.train_loss_opt,
            "test_loss_opt": self.test_loss_opt,

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
            y_pred_train_init=np.array(body["y_pred_train_init"]),
            y_pred_train_opt=np.array(body["y_pred_train_opt"]),

            x_test=np.array(body["x_test"]),
            y_test=np.array(body["y_test"]),
            y_pred_test_init=np.array(body["y_pred_test_init"]),
            y_pred_test_opt=np.array(body["y_pred_test_opt"]),

            theta_init=np.array(body["theta_init"]),
            x_plot_init=np.array(body["x_plot_init"]),
            y_plot_init=np.array(body["y_plot_init"]),
            train_loss_init=body["train_loss_init"],
            test_loss_init=body["test_loss_init"],

            theta_opt=np.array(body["theta_opt"]),
            x_plot_opt=np.array(body["x_plot_opt"]),
            y_plot_opt=np.array(body["y_plot_opt"]),
            train_loss_opt=body["train_loss_opt"],
            test_loss_opt=body["test_loss_opt"],

            iteration_records=[IterationRecord.from_json(x) for x in body["iteration_records"]],
        )

    @property
    def opt_maxiter(self):
        m = re.search(r"\bmaxiter=(\d+)\b", self.opt_options)
        if m:
            return int(m.group(1))
        else:
            return None
