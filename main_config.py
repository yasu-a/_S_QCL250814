import json
import os
from dataclasses import dataclass
from typing import Literal

import joblib

import repo
from generate_random_seed import generate_random_seed
from repo import RecordKey
from run import run
from run_param import RunParam


@dataclass(frozen=True)
class SessionEntry:
    n_run: int
    n_qubit: int
    func_type: Literal["gauss10", "gauss5", "gauss3", "tri"]
    method: Literal["conv", "pow_3", "nonint"]
    opt_method: Literal["Nelder-Mead", "SLSQP", "BFGS"]

    def __post_init__(self):
        if self.n_run <= 0:
            raise ValueError(f"n_run must be positive: {self.n_run}")
        if self.n_qubit <= 0:
            raise ValueError(f"n_qubit must be positive: {self.n_qubit}")
        if self.func_type not in ["gauss10", "gauss5", "gauss3", "tri"]:
            raise ValueError(
                f"func_type must be one of 'gauss10', 'gauss5', 'gauss3', 'tri': {self.func_type}")
        if self.method not in ["conv", "pow_3", "nonint"]:
            raise ValueError(
                f"method must be one of 'conv', 'pow_3', 'nonint': {self.method}")
        if self.opt_method not in ["Nelder-Mead", "SLSQP", "BFGS"]:
            raise ValueError(
                f"opt_method must be one of 'Nelder-Mead', 'SLSQP', 'BFGS': {self.opt_method}")

    @classmethod
    def from_json(cls, body):
        return cls(
            n_run=body["n_run"],
            n_qubit=body["n_qubit"],
            func_type=body["func_type"],
            method=body["method"],
            opt_method=body["opt_method"],
        )

    def to_record_key(self) -> RecordKey:
        return RecordKey(
            nqubit=self.n_qubit,
            func_type=self.func_type,
            method=self.method,
            opt_method=self.opt_method,
        )


@dataclass(frozen=True)
class GlobalConfig:
    n_cpu: int
    max_iter: int
    abs_tol: float

    def __post_init__(self):
        if not isinstance(self.n_cpu, int) or self.n_cpu < -2:
            raise ValueError(
                f"n_cpu must be positive integer, -1 (use all CPUs), or -2 (use all CPUs except one): {self.n_cpu}")
        if self.n_cpu > os.cpu_count():
            raise ValueError(
                f"n_cpu must be less than or equal to the number of CPUs: n_cpu={self.n_cpu}, {os.cpu_count()=}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive: {self.max_iter}")
        if self.abs_tol <= 0:
            raise ValueError(f"abs_tol must be positive: {self.abs_tol}")

    @classmethod
    def from_json(cls, body):
        return cls(
            n_cpu=body["n_cpu"],
            max_iter=body["max_iter"],
            abs_tol=body["abs_tol"],
        )

    @property
    def n_cpu_actual(self) -> int:
        if self.n_cpu == -1:
            n_cpu_actual = os.cpu_count()
        elif self.n_cpu == -2:
            n_cpu_actual = os.cpu_count() - 1
        else:
            n_cpu_actual = self.n_cpu
        return max(min(n_cpu_actual, os.cpu_count()), 1)


@dataclass(frozen=True)
class ConfigJson:
    global_config: GlobalConfig
    sessions: list[SessionEntry]

    def __post_init__(self):
        if not isinstance(self.global_config, GlobalConfig):
            raise ValueError(
                f"global_config must be an instance of GlobalConfig: {self.global_config}")
        if not isinstance(self.sessions, list):
            raise ValueError(f"sessions must be a list: {self.sessions}")
        if not all(isinstance(session, SessionEntry) for session in self.sessions):
            raise ValueError(
                f"sessions must be a list of SessionEntry: {self.sessions}")
        # check sessions are unique
        if len(self.sessions) != len(set(self.sessions)):
            raise ValueError("sessions must be unique")

    @classmethod
    def from_json(cls, body):
        global_config = GlobalConfig.from_json(body["global_config"])
        sessions = []
        for session in body["sessions"]:
            if isinstance(session, dict):
                sessions.append(SessionEntry.from_json(session))
            elif isinstance(session, list):
                if len(session) != 5:
                    raise ValueError(
                        f"Session must be a list of 5 elements: "
                        f"opt_method, n_run, n_qubit, func_type, method"
                    )
                opt_method, n_run, n_qubit, func_type, method = session
                sessions.append(
                    SessionEntry(
                        opt_method=opt_method,
                        n_run=n_run,
                        n_qubit=n_qubit,
                        func_type=func_type,
                        method=method,
                    )
                )
            else:
                raise ValueError(f"Invalid session type: {type(session)}")
        return cls(
            global_config=global_config,
            sessions=sessions,
        )


def main(config_json_path: str):
    # parse config json
    with open(config_json_path, "r", encoding="utf-8") as f:
        try:
            config = ConfigJson.from_json(json.load(f))
        except Exception as e:
            raise ValueError(
                f"Failed to load config json: \"{config_json_path}\". {type(e).__name__!s}: {e}")
        print(config)

    # create run parameters
    run_params: list[RunParam] = []
    for session in config.sessions:
        # find number of tasks left in the session
        n_run_current = repo.count_records(session.to_record_key())
        n_run_required = session.n_run
        if n_run_current > n_run_required:
            print(
                f"Warning: {session.to_record_key()} has {n_run_current} records, but {n_run_required} are required")
            n_run_required = n_run_current
        n_rest = n_run_required - n_run_current

        # generate run params
        for _ in range(n_rest):
            run_params.append(
                RunParam(
                    nqubit=session.n_qubit,
                    func_type=session.func_type,
                    method=session.method,
                    opt_method=session.opt_method,
                    max_iter=config.global_config.max_iter,
                    abs_tol=config.global_config.abs_tol,
                    seed_system=generate_random_seed(),
                    seed_time_evol=generate_random_seed(),
                    seed_theta_init=generate_random_seed(),
                )
            )

    # run with joblib
    with joblib.Parallel(n_jobs=config.global_config.n_cpu_actual) as parallel:
        parallel(joblib.delayed(run)(run_param) for run_param in run_params)


if __name__ == '__main__':
    main(config_json_path="config.json")
