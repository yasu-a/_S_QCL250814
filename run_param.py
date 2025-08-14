from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunParam:
    """run関数のパラメータを格納するデータクラス"""
    # 必須パラメータ
    nqubit: int  # qubit数
    func_type: Literal["gauss10", "gauss5", "gauss3", "tri"]  # 学習する関数の種類（ガウス分布，三角波）
    method: Literal["conv", "pow_3", "nonint"]  # データアップロードの手法（従来手法＝均一回転角，3のi乗，非可積分系）
    opt_method: Literal["Nelder-Mead", "BFGS", "SLSQP"]  # 最適化手法
    max_iter: int  # 最適化の最大イテレーション回数
    abs_tol: float  # 最適化でコストがこれ以下になったら中断
    seed_system: int  # ハミルトニアンに含まれる係数を作るためのシード（非可積分系でしか使われない）
    seed_time_evol: int  # 時間発展演算子を作るためのシード
    seed_theta_init: int  # パラメータthetaを作るためのシード

    # オプションパラメータ（デフォルト値付き）
    c_depth: int = 3
    time_step: float = 0.77
    obs_coeff: float = 2.0
    seed_train: int = 0

    def __post_init__(self) -> None:
        """パラメータのタイプチェッキングとバリデーション"""
        # タイプチェック
        assert isinstance(self.nqubit, int), \
            f"nqubit must be int, got {type(self.nqubit)}"
        assert isinstance(self.c_depth, int), \
            f"c_depth must be int, got {type(self.c_depth)}"
        assert isinstance(self.time_step, (float, int)), \
            f"time_step must be float or int, got {type(self.time_step)}"
        assert isinstance(self.obs_coeff, (float, int)), \
            f"obs_coeff must be float or int, got {type(self.obs_coeff)}"
        assert isinstance(self.func_type, str), \
            f"func_type must be str, got {type(self.func_type)}"
        assert isinstance(self.method, str), \
            f"method must be str, got {type(self.method)}"
        assert isinstance(self.opt_method, str), \
            f"opt_method must be str, got {type(self.opt_method)}"
        assert isinstance(self.max_iter, int), \
            f"max_iter must be int, got {type(self.max_iter)}"
        assert isinstance(self.abs_tol, (float, int)), \
            f"abs_tol must be float or int, got {type(self.abs_tol)}"
        assert isinstance(self.seed_train, int), \
            f"seed_train must be int, got {type(self.seed_train)}"
        assert isinstance(self.seed_system, int), \
            f"seed_system must be int, got {type(self.seed_system)}"
        assert isinstance(self.seed_time_evol, int), \
            f"seed_time_evol must be int, got {type(self.seed_time_evol)}"
        assert isinstance(self.seed_theta_init, int), \
            f"seed_theta_init must be int, got {type(self.seed_theta_init)}"

        # 値の範囲チェック
        assert self.nqubit > 0, \
            f"nqubit must be positive, got {self.nqubit}"
        assert self.c_depth > 0, \
            f"c_depth must be positive, got {self.c_depth}"
        assert self.time_step > 0, \
            f"time_step must be positive, got {self.time_step}"
        assert self.max_iter > 0, \
            f"max_iter must be positive, got {self.max_iter}"
        assert self.abs_tol > 0, \
            f"abs_tol must be positive, got {self.abs_tol}"

        # 選択肢の検証
        valid_func_types = ["gauss10", "gauss5", "gauss3", "tri"]
        assert self.func_type in valid_func_types, \
            f"func_type must be one of {valid_func_types}, got {self.func_type}"

        valid_methods = ["conv", "pow_3", "nonint"]
        assert self.method in valid_methods, \
            f"method must be one of {valid_methods}, got {self.method}"

        valid_opt_methods = ["Nelder-Mead", "BFGS", "SLSQP"]
        assert self.opt_method in valid_opt_methods, \
            f"opt_method must be one of {valid_opt_methods}, got {self.opt_method}"
