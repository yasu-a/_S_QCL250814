import datetime
import json
import os
import time
from contextlib import redirect_stdout
from functools import lru_cache, reduce, wraps
from typing import Callable

import numpy as np
import qulacs
from qulacs import QuantumCircuit, QuantumState, Observable
from scipy.optimize import OptimizeResult, minimize
from tqdm import tqdm

from model.run_param import RunParam
from model.run_record import IterationRecord, GlobalRecord
from model.value import FuncType, MethodType, OptMethodType
from repo import put_record
from util import X_mat, Y_mat, Z_mat, fullgate, ham_to_gate_mat, \
    gate_mat_to_gate, ham_to_gate, make_fullgate, CacheSession

"""
回帰
ノイズ入れろstd 適当

train test split
 - train loss 200
 - test loss 200
 trainと同様にtestの箱ひげを出す

classification
 - 階段関数(中心は0）をやる
 - train/test 200/200
 cross entropy
 
testのスコアが最小のところを使う
"""


class Dataset:
    def __init__(
            self,
            *,
            func_to_learn: Callable[[np.ndarray], np.ndarray],
            x_min: float,
            x_max: float,
            noise_std: float,
            seed: int,
    ):
        self._func_to_learn = func_to_learn
        self._x_min = x_min
        self._x_max = x_max
        self._noise_std = noise_std

        self._rng = np.random.RandomState(seed)

    def x_seq(self, n: int) -> np.ndarray:
        return np.linspace(self._x_min, self._x_max, n)

    @classmethod
    def create_from_func_type(
            cls,
            *,
            func_type: FuncType,
            x_min: float,
            x_max: float,
            noise_std: float,
            seed: int,
    ) -> "Dataset":
        if func_type == "gauss10":
            def func_to_learn(x):
                return np.exp(-10 * x ** 2)
        elif func_type == "gauss5":
            def func_to_learn(x):
                return np.exp(-5 * x ** 2)
        elif func_type == "gauss3":
            def func_to_learn(x):
                return np.exp(-3 * x ** 2)
        elif func_type == "tri":
            def func_to_learn(x):
                return np.select(
                    condlist=[(-1 <= x) & (x <= -0.5), (-0.5 < x)
                              & (x <= 0.5), (0.5 < x) & (x <= 1)],
                    choicelist=[-2 * (x + 1), 2 * x, -2 * (x - 1)],
                )
        else:
            assert False, func_type
        return cls(
            func_to_learn=func_to_learn,
            x_min=x_min,
            x_max=x_max,
            noise_std=noise_std,
            seed=seed
        )

    def create_data(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = self._rng.uniform(self._x_min, self._x_max, n)
        y = self._func_to_learn(x)
        y += self._rng.normal(loc=0, scale=self._noise_std, size=n)
        return x, y


def wrap_process(runner: Callable[[RunParam], None]) -> Callable[[RunParam], None]:
    # g.redirect_output_to_log_file is True なら stdout をファイルにリダイレクトする
    @wraps(runner)
    def wrapper(g: RunParam) -> None:
        if g.redirect_output_to_log_file:
            os.makedirs("./log", exist_ok=True)
            log_file_path = f'./log/output_{abs(hash(g))}.log'
            print(
                f" *** Redirect stdout of following session to \"{log_file_path}\"")
            print(g)
            with open(log_file_path, 'w') as f, redirect_stdout(f):
                return runner(g)
        else:
            return runner(g)

    return wrapper


def create_u_in(
        *,
        seed: int,
        n_qubit: int,
        time_step: float,
        method: MethodType,
        cache_session: CacheSession,
) -> Callable[[float], qulacs.QuantumCircuit]:
    rng = np.random.RandomState(seed)

    if method == "conv":
        # 確認済み
        @cache_session.attach_cache(lru_cache(maxsize=1024))
        def U_in_mat(x: float) -> np.ndarray:
            ham = np.eye(2 ** n_qubit, dtype=complex)
            for i in reversed(range(n_qubit)):
                mat = fullgate(n_qubit, f"Y{i}") * x / 2 / -time_step
                ham @= ham_to_gate_mat(mat, time_step)
            return ham

        # # 確認済み
        # def U_in(x):
        #     U = QuantumCircuit(n_qubit)

        #     for i in range(n_qubit):
        #         U.add_RY_gate(i, x)

        #     return U

    elif method == "pow_3":
        # 確認済み
        @cache_session.attach_cache(lru_cache(maxsize=1024))
        def U_in_mat(x: float) -> np.ndarray:
            ham = np.eye(2 ** n_qubit, dtype=complex)
            for i in reversed(range(n_qubit)):
                mat = fullgate(n_qubit, f"Y{i}") * (
                        x * 3 ** (n_qubit - 1 - i)) / 2 / -time_step  # Qulacsはビット番号が逆
                ham @= ham_to_gate_mat(mat, time_step)
            return ham

        # # 確認済み
        # def U_in(x):
        #     U = QuantumCircuit(n_qubit)

        #     for i in range(n_qubit):
        #         U.add_RY_gate(i, x * 3 ** i)

        #     return U

    elif method == "nonint":
        Bx_list = rng.uniform(-1, 1, n_qubit)
        By_list = rng.uniform(-1, 1, n_qubit)
        Bz_list = rng.uniform(-1, 1, n_qubit)
        Jmat = np.zeros((n_qubit, n_qubit))
        for i in range(n_qubit):
            for j in range(n_qubit):
                Jmat[i][j] = rng.uniform(-1, 1)

        # print("Bx_list")
        # print(Bx_list)
        # print()
        # print("By_list")
        # print(By_list)
        # print()
        # print("Bz_list")
        # print(Bz_list)
        # print()
        # print("Jmat")
        # print(Jmat)
        # print()

        @cache_session.attach_cache(lru_cache(maxsize=1024))
        def U_in_mat(x: float) -> np.ndarray:
            dataham = np.zeros((2 ** n_qubit, 2 ** n_qubit), dtype=complex)

            for i in range(n_qubit):
                B = Bx_list[i]
                dataham += B * fullgate(n_qubit, f"X{i}")
                B = By_list[i]
                dataham += B * fullgate(n_qubit, f"Y{i}")
                B = Bz_list[i]
                dataham += B * fullgate(n_qubit, f"Z{i}")

            for i in range(n_qubit - 1):
                for j in range(i + 1, n_qubit):
                    dataham += Jmat[i][j] * \
                               make_fullgate([[i, X_mat], [j, X_mat]], n_qubit)
                    dataham += Jmat[i][j] * \
                               make_fullgate([[i, Y_mat], [j, Y_mat]], n_qubit)
                    dataham += 0.73 * \
                               Jmat[i][j] * \
                               make_fullgate([[i, Z_mat], [j, Z_mat]], n_qubit)

            return ham_to_gate_mat(dataham, x)  # exp(-iHt) ではなく exp(-iHx) の形

    def U_in(x) -> qulacs.QuantumCircuit:
        U = QuantumCircuit(n_qubit)
        mat = U_in_mat(x)
        gate = gate_mat_to_gate(mat)
        U.add_gate(gate)
        return U

    return U_in


def create_time_evol_gate(*, seed: int, n_qubit: int, time_step: float) -> qulacs.QuantumGateMatrix:
    # ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
    rng = np.random.RandomState(seed)

    ham = np.zeros((2 ** n_qubit, 2 ** n_qubit), dtype=complex)
    for i in range(n_qubit):  # i runs 0 to n_qubit-1
        Jx = -1. + 2. * rng.rand()  # -1~1の乱数
        ham += Jx * fullgate(n_qubit, f"X{i}")
        for j in range(i + 1, n_qubit):
            J_ij = -1. + 2. * rng.rand()
            ham += J_ij * fullgate(n_qubit, f"Z{i},Z{j}")

    # 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
    time_evol_gate = ham_to_gate(ham, time_step)

    return time_evol_gate


# MSE（二乗平均平方誤差）を計算する関数
def mse(*, y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def fit(
        *,
        x_train, y_train,
        x_val, y_val,
        target_func_factory: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], float]],
        theta_init: np.ndarray,
        cache_session: CacheSession,
        show_progress_bar: bool,
        opt_method: OptMethodType,
        max_iter: int,
        abs_tol: float,
        # アーリーストッピング用パラメータ
        early_stopping_enabled: bool,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
):
    # opt_method = "BFGS" # @param ["Nelder-Mead", "SLSQP", "BFGS"]
    # max_iter = 100  # @param {type: "integer"}
    # abs_tol = 1e-5  # @param {type: "number"}  # コストがこれ以下になったら中断
    eps = 1e-10  # @param {type: "number"}  # 微分の近似に使う小さな値

    n_iter = 0  # callbackで使うイテレーション回数のカウンタ
    time_start = time.monotonic()  # 開始時刻
    time_info = time.monotonic()  # 最後に最適化の状態を表示した時刻
    if show_progress_bar:
        pbar = tqdm(total=max_iter)  # プログレスバー
    else:
        pbar = None

    # 最適化の履歴を記録するリスト
    loss_history = []

    # アーリーストッピング用の変数
    best_val_loss = float('inf')  # これまでの最小検証損失
    patience_counter = 0  # 検証損失が改善されなかったイテレーション数
    is_early_stopped = False  # アーリーストッピングが発生したかどうか

    # 最初の状態を追加
    train_loss_init = target_func_factory(x_train, y_train)(theta_init)
    val_loss = None
    if early_stopping_enabled:
        # 検証損失を計算
        val_loss = target_func_factory(x_val, y_val)(theta_init)
    record = {
        "n_iter": 0,
        "train_loss": train_loss_init,
        "theta": theta_init,
        "elapsed_time": 0,
    }
    if val_loss is not None:  # early_stopping_enabled is True
        record["val_loss"] = val_loss
        record["best_val_loss"] = best_val_loss
        record["patience_counter"] = patience_counter
    loss_history.append(record)

    # コールバック関数を定義（松崎変更）→計算途中にコスト関数が見えるようにした
    def common_callback(n_iter: int, theta: np.ndarray, current_train_loss: float):
        nonlocal time_info, best_val_loss, patience_counter, is_early_stopped

        # 現在時刻を取得
        time_now = time.monotonic()
        elapsed_time = time_now - time_start  # 経過時間

        # アーリーストッピングのチェック
        val_loss = None
        if early_stopping_enabled:
            # 検証損失を計算
            val_loss = target_func_factory(x_val, y_val)(theta)

            # 検証損失が改善されたかチェック
            if val_loss < best_val_loss - early_stopping_min_delta:
                # 改善があった場合
                best_val_loss = val_loss
                patience_counter = 0
            else:
                # 改善がなかった場合
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

                # patienceに達した場合、アーリーストッピング
                if patience_counter >= early_stopping_patience:
                    is_early_stopped = True
                    print(f"Early stopping triggered after {n_iter} iterations")
                    print(f"Best validation loss: {best_val_loss:.7f}")

        # 履歴の記録
        record = {
            "n_iter": n_iter,
            "train_loss": current_train_loss,
            "theta": theta,
            "elapsed_time": elapsed_time,
        }
        if val_loss is not None:  # early_stopping_enabled is True
            record["val_loss"] = val_loss
            record["best_val_loss"] = best_val_loss
            record["patience_counter"] = patience_counter
        loss_history.append(record)

        # プログレスバーの更新
        if pbar is not None:
            pbar.update()
            desc = f"Iteration #{n_iter}, train_loss={current_train_loss:.7f}"
            if val_loss is not None:
                desc += f", val_loss={val_loss:.7f}, patience={patience_counter}/{early_stopping_patience}"
            pbar.set_description(desc)

        # 現在の状況の表示
        if time_now - time_info >= 30:  # 30秒に1回だけ状況を表示
            time_info = time_now
            with np.printoptions(precision=3, suppress=True):
                info_str = f"Iteration #{n_iter}({datetime.timedelta(seconds=elapsed_time)}), train_loss={current_train_loss:.7f}"
                if val_loss is not None:
                    info_str += f", val_loss={val_loss:.7f}, best_val_loss={best_val_loss:.7f}, patience={patience_counter}/{early_stopping_patience}"
                print(info_str)
                print(f"Current theta:")
                print(np.array2string(theta, separator=", ", max_line_width=150))
            cache_session.print_cache_profile()
            print()

    if opt_method == "Nelder-Mead":
        def callback(intermediate_result: OptimizeResult):
            nonlocal n_iter
            theta = intermediate_result.x
            n_iter += 1  # カウントの更新
            current_train_loss = intermediate_result.fun  # 現在のコストを計算
            common_callback(n_iter, theta, current_train_loss)

            # アーリーストッピングまたはabs_tol到達で停止
            if is_early_stopped:
                raise StopIteration("Early stopping")
            if current_train_loss < abs_tol:
                raise StopIteration("Absolute tolerance reached")

        minimize_kwargs = dict(  # minimizeに与えるパラメータをプロットに印字するために分離
            method='Nelder-Mead',
            callback=callback,
            options={
                'disp': True,
                'maxiter': max_iter,
            },
        )
    elif opt_method == "SLSQP":
        assert not early_stopping_enabled, "SLSQP does not support early stopping"

        def callback(theta):
            nonlocal n_iter
            n_iter += 1  # カウントの更新
            current_train_loss = target_func_factory(x_train, y_train)(theta)  # 現在のコストを計算
            common_callback(n_iter, theta, current_train_loss)

        minimize_kwargs = dict(  # minimizeに与えるパラメータをプロットに印字するために分離
            method='SLSQP',
            callback=callback,
            jac=None,
            options={
                'disp': True,
                'maxiter': max_iter,
                'ftol': abs_tol,
                'eps': eps,
            },
        )
    elif opt_method == "BFGS":
        # コールバック関数を定義（松崎変更）→計算途中にコスト関数が見えるようにした
        def callback(intermediate_result: OptimizeResult):
            nonlocal n_iter
            theta = intermediate_result.x
            n_iter += 1  # カウントの更新
            current_train_loss = intermediate_result.fun  # 現在のコストを計算
            common_callback(n_iter, theta, current_train_loss)

            # アーリーストッピングまたはabs_tol到達で停止
            if is_early_stopped:
                raise StopIteration("Early stopping")
            if current_train_loss < abs_tol:
                raise StopIteration("Absolute tolerance reached")

        minimize_kwargs = dict(  # minimizeに与えるパラメータをプロットに印字するために分離
            method="BFGS",
            callback=callback,
            jac=None,
            options={
                'disp': True,
                'maxiter': max_iter,
                'eps': eps,
            },
        )
    else:
        assert False, f"invalid method '{opt_method}'"

    # minimize_kwargsの文字列化
    minimize_kwargs_str = ", ".join(
        f"{k}={v}"
        for k, v in (minimize_kwargs | minimize_kwargs["options"]).items()
        if k not in {"callback", "options", "disp"}
    )

    print(" *** Optimizer options")
    print(minimize_kwargs_str)
    print()

    # 学習 (筆者のPCで1~2分程度かかる)（松崎変更）callbackを追加
    try:
        minimize(
            target_func_factory(x_train, y_train),
            theta_init,
            **minimize_kwargs,  # ここでまとめて与える
        )
    except KeyboardInterrupt:
        is_intermediate_result = True
    else:
        is_intermediate_result = False

    return minimize_kwargs_str, loss_history, is_intermediate_result, is_early_stopped


def create_u_out(
        *,
        cache_session: CacheSession,
        c_depth: int,
        n_qubit: int,
        time_step: float,
        time_evol_gate: qulacs.QuantumGateMatrix,
) -> Callable[[np.ndarray], qulacs.QuantumCircuit]:
    @cache_session.attach_cache(lru_cache(maxsize=1024))
    def U_out_mat(theta: np.ndarray) -> np.ndarray:
        theta = theta.reshape(c_depth, n_qubit, 3)[:, ::-1, :].flatten()
        # ^ qulacs.ParametricQuantumCircuitはnqubitに渡って逆順にパラメータを渡しているので整合性をとる
        theta_it = iter(theta)

        gate_mat_lst = []
        for d in range(c_depth):
            gate_mat_lst.append(time_evol_gate.get_matrix())
            for i in range(n_qubit):
                gate_mat_lst.append(
                    ham_to_gate_mat(
                        fullgate(n_qubit, f"X{i}") * next(theta_it) / 2 / -time_step,
                        time_step,
                    )
                )
                gate_mat_lst.append(
                    ham_to_gate_mat(
                        fullgate(n_qubit, f"Z{i}") * next(theta_it) / 2 / -time_step,
                        time_step,
                    )
                )
                gate_mat_lst.append(
                    ham_to_gate_mat(
                        fullgate(n_qubit, f"X{i}") * next(theta_it) / 2 / -time_step,
                        time_step,
                    )
                )

        try:
            next(theta_it)
        except StopIteration:
            pass
        else:
            assert False, "thetaが全て消費されていません"

        total_gate_mat = reduce(np.matmul, reversed(gate_mat_lst))

        return total_gate_mat

    def U_out(theta: np.ndarray) -> QuantumCircuit:
        U = QuantumCircuit(n_qubit)
        mat = U_out_mat(theta)
        gate = gate_mat_to_gate(mat)
        U.add_gate(gate)
        return U

    return U_out


@wrap_process
def run(g: RunParam) -> None:
    cache_session = CacheSession()

    # データの生成
    ds = Dataset.create_from_func_type(
        func_type=g.func_type,
        x_min=-1.0,
        x_max=+1.0,
        noise_std=g.dataset_noise_std,
        seed=g.seed_train,
    )
    x_train, y_train = ds.create_data(g.n_train)
    x_test, y_test = ds.create_data(g.n_test)

    # U_in
    U_in = create_u_in(
        seed=g.seed_system,
        n_qubit=g.nqubit,
        method=g.method,
        time_step=g.time_step,
        cache_session=cache_session,
    )

    # 確認用コード
    print(f"{g.method=}")
    s = QuantumState(g.nqubit)
    s.set_zero_state()
    U_in(0.1).update_quantum_state(s)
    with np.printoptions(precision=3, suppress=True, linewidth=70):
        print(np.array2string(s.get_vector(), separator=","))

    # ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
    time_evol_gate = create_time_evol_gate(
        seed=g.seed_time_evol,
        n_qubit=g.nqubit,
        time_step=g.time_step,
    )

    # 確認用コード
    with np.printoptions(suppress=True, precision=3, linewidth=999):
        print(time_evol_gate.get_matrix())

    # U_out
    U_out = create_u_out(
        cache_session=cache_session,
        c_depth=g.c_depth,
        n_qubit=g.nqubit,
        time_step=g.time_step,
        time_evol_gate=time_evol_gate,
    )

    # パラメータ数
    parameter_count = 3 * g.nqubit * g.c_depth

    # 確認用コード
    print(f"{g.method=}")
    rng = np.random.RandomState(9999)
    theta_init = rng.uniform(0, 2 * np.pi, parameter_count)
    s = QuantumState(g.nqubit)
    s.set_zero_state()
    U_in(0.1).update_quantum_state(s)
    print("s_in")
    with np.printoptions(precision=3, suppress=True, linewidth=70):
        print(s.get_vector())
    U_out(theta_init).update_quantum_state(s)
    print("s_out")
    with np.printoptions(precision=3, suppress=True, linewidth=70):
        print(s.get_vector())

    # オブザーバブルZ_0を作成
    obs = Observable(g.nqubit)
    obs.add_operator(g.obs_coeff, 'Z 0')

    # 入力x_iからモデルの予測値y(x_i, theta)を返す関数
    def qcl_pred(x, theta):
        state = QuantumState(g.nqubit)
        state.set_zero_state()

        # 入力状態計算
        U_in(x).update_quantum_state(state)

        # 出力状態計算
        U_out(theta).update_quantum_state(state)

        # モデルの出力
        res = obs.get_expectation_value(state)

        return res

    # cost function Lを計算
    def cost_factory(x_data: np.ndarray, y_data: np.ndarray):
        def cost(theta):
            y_pred = np.array([qcl_pred(x, theta) for x in x_data])
            loss = mse(y_pred=y_pred, y_true=y_data)
            return loss

        return cost

    """# 学習"""

    # seed_theta_init = 123  # @param {type: "integer"}
    rng = np.random.RandomState(g.seed_theta_init)
    theta_init = rng.uniform(0, 2 * np.pi, parameter_count)

    # パラメータthetaの初期値のもとでのグラフ
    print(f"{g.method=}")
    xlist = ds.x_seq(n=100)
    # y_init = [qcl_pred(x, theta_init) for x in xlist]
    # plt.plot(xlist, y_init)

    system_setup_dct = {
        "nqubit": g.nqubit,
        "c_depth": g.c_depth,
        "time_step": g.time_step,
        "obs_coeff": g.obs_coeff,
        "func_type": g.func_type,
        "method": g.method,
        "seed_train": g.seed_train,
        "seed_system": g.seed_system,
        "seed_time_evol": g.seed_time_evol,
        "seed_theta_init": g.seed_theta_init,
    }
    system_setup_str = json.dumps(system_setup_dct)

    print(" *** System setup")
    print(system_setup_str)
    print()

    minimize_kwargs_str, loss_history, is_intermediate_result, is_early_stopped = fit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test,
        target_func_factory=cost_factory,
        theta_init=theta_init,
        cache_session=cache_session,
        show_progress_bar=g.show_progress_bar,
        opt_method=g.opt_method,
        max_iter=g.max_iter,
        abs_tol=g.abs_tol,
        # アーリーストッピング用パラメータ
        early_stopping_enabled=g.early_stopping_enabled,
        early_stopping_patience=g.early_stopping_patience,
        early_stopping_min_delta=g.early_stopping_min_delta,
    )
    train_loss_init = loss_history[0]["train_loss"]
    test_loss_init = loss_history[0]["val_loss"]
    # 実行が中断されてもひとまず途中までの結果を最適化の結果とする
    train_loss_opt = loss_history[-1]["train_loss"]
    test_loss_opt = loss_history[-1]["val_loss"]
    theta_opt = loss_history[-1]["theta"]

    # # 履歴のdataframeを作る
    # df_loss_history = pd.DataFrame(loss_history)
    # df_loss_history["elapsed_time_str"] \
    #     = df_loss_history["elapsed_time"].map(lambda x: f"\n{datetime.timedelta(seconds=int(x))}")

    if is_intermediate_result:
        for _ in range(5):
            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! OPTIMIZATION INTERRUPTED, NOT FINAL RESULT !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print()
    elif is_early_stopped:
        for _ in range(3):
            print()
            print("*** EARLY STOPPING ACTIVATED ***")
            print(
                f"*** Training stopped after {len(loss_history) - 1} iterations due to no improvement in validation loss ***")
            print()
    print()
    print(" *** System setup")
    print(system_setup_str)
    print()
    print(" *** Optimizer options")
    print(minimize_kwargs_str)
    print()
    print(f" *** theta_init ({train_loss_init=:.7f})")
    with np.printoptions(precision=7, suppress=True):
        print(theta_init)
    print()
    print(f" *** theta_opt ({train_loss_opt=:.7f})")
    with np.printoptions(precision=7, suppress=True):
        print(theta_opt)

    # # プロット
    # plt.figure(figsize=(10, 6))

    # xlist = np.arange(x_min, x_max, 0.02)

    # # 教師データ
    # plt.plot(x_train, y_train, "o", label='Teacher')

    # # パラメータθの初期値のもとでのグラフ
    # plt.plot(xlist, y_init, '--', label='Initial Model Prediction', c='gray')

    # # モデルの予測値
    # y_pred = np.array([qcl_pred(x, theta_opt) for x in xlist])
    # plt.plot(xlist, y_pred, label='Final Model Prediction')

    # plt.ylim(-1.1, 1.1)
    # plt.grid()
    # plt.legend()
    # plt.show()

    if not is_intermediate_result:  # 中断されていなければ保存（アーリーストッピングは含む）
        iteration_records = [
            IterationRecord(
                n_iter=h["n_iter"],
                elapsed_time=h["elapsed_time"],
                train_loss=float(h["train_loss"]),  # np.float64 -> float
                val_loss=float(h["val_loss"]),  # np.float64 -> float
                best_val_loss=float(h["best_val_loss"]),  # np.float64 -> float
                patience_counter=int(h["patience_counter"]),  # np.int64 -> int
            )
            for h in loss_history
        ]
        global_record = GlobalRecord(
            nqubit=g.nqubit,
            c_depth=g.c_depth,
            time_step=g.time_step,

            obs_coeff=g.obs_coeff,

            func_type=g.func_type,
            method=g.method,

            opt_method=g.opt_method,
            opt_options=minimize_kwargs_str,

            seed_train=g.seed_train,
            seed_system=g.seed_system,
            seed_time_evol=g.seed_time_evol,
            seed_theta_init=g.seed_theta_init,

            x_train=x_train,
            y_train=y_train,
            y_pred_train_init=np.array([qcl_pred(x, theta_init) for x in x_train]),
            y_pred_train_opt=np.array([qcl_pred(x, theta_opt) for x in x_train]),

            x_test=x_test,
            y_test=y_test,
            y_pred_test_init=np.array([qcl_pred(x, theta_init) for x in x_test]),
            y_pred_test_opt=np.array([qcl_pred(x, theta_opt) for x in x_test]),

            theta_init=theta_init,
            x_plot_init=xlist,
            y_plot_init=np.array([qcl_pred(x, theta_init) for x in xlist]),
            train_loss_init=train_loss_init,
            test_loss_init=test_loss_init,

            theta_opt=theta_opt,
            x_plot_opt=xlist,
            y_plot_opt=np.array([qcl_pred(x, theta_opt) for x in xlist]),
            train_loss_opt=train_loss_opt,
            test_loss_opt=test_loss_opt,

            iteration_records=iteration_records,
        )
        put_record(global_record)
