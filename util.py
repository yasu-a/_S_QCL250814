import math
from functools import wraps, reduce, cache
from typing import Callable, Any

import numpy as np
import pandas as pd
import qulacs
from qulacs.gate import X, Y, Z, H

I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Y_mat = Y(0).get_matrix()
Z_mat = Z(0).get_matrix()
H_mat = H(0).get_matrix()


# fullsizeのgateをつくる関数.
def make_fullgate(list_SiteAndOperator, nqubit):
    '''
    list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
    関係ないqubitにIdentityを挿入して
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    という(2**nqubit, 2**nqubit)行列をつくる.
    '''
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []  ## 1-qubit gateを並べてnp.kronでreduceする
    cnt = 0
    for i in range(nqubit):
        if (i in list_Site):
            list_SingleGates.append(list_SiteAndOperator[cnt][1])
            cnt += 1
        else:  ## 何もないsiteはidentity
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)


class CacheSession:
    def __init__(self):
        self._cached_func_registry = []

    @classmethod
    def returns_readonly_array(cls, f: Callable[[Any], np.ndarray]):
        """ NumPy配列を返す関数の出力を読み取り専用にするデコレータ

        このデコレータを適用すると、関数が返すNumPy配列の書き込みフラグがFalseに設定され、
        意図しない変更を防ぐ

        Args:
            f: NumPy配列を返す関数。

        Returns:
            ラップされた関数。この関数は読み取り専用のNumPy配列を返します。
        """

        @wraps(f)
        def wrapper(*args, **kwargs) -> np.ndarray:
            result = f(*args, **kwargs)
            result.setflags(write=False)
            return result

        return wrapper

    def _register_for_cache_profile(self, cached_func):
        """
        キャッシュプロファイリングのためにキャッシュされた関数を内部レジストリに登録します。

        この関数は `attach_cache` デコレータ内で使用され、キャッシュされた関数の情報を追跡します。
        ユーザーが直接呼び出すことは想定されていません。

        Args:
            cached_func: キャッシュされた関数オブジェクト。
        """

        self._cached_func_registry.append(cached_func)

    def print_cache_profile(self) -> None:
        """
        登録されたキャッシュされた関数のプロファイル情報を表示します。

        この関数は、キャッシュのヒット数、ミス数、最大サイズ、現在のサイズなどの統計情報を
        pandas DataFrame形式で出力します。これにより、キャッシュの利用状況を把握し、
        パフォーマンスチューニングに役立てることができます。
        """

        print("=" * 40, "CACHE PROFILE", "=" * 40)
        data = []
        for cached_func in self._cached_func_registry:
            data.append({
                "name": cached_func.__name__,
                "hits": cached_func.cache_info().hits,
                "misses": cached_func.cache_info().misses,
                "maxsize": cached_func.cache_info().maxsize,
                "currsize": cached_func.cache_info().currsize,
            })
        with pd.option_context(
                'display.max_rows', 999, 'display.max_columns', 999, 'display.width', 999
        ):
            print(pd.DataFrame(data))
        print("=" * (40 + 1 + len("CACHE PROFILE") + 1 + 40))

    def attach_cache(self, cache_decorator):
        """
        指定されたキャッシュデコレータを適用し、キャッシュプロファイリングのために登録するデコレータを返します。

        このデコレータは、関数をキャッシュするだけでなく、その関数のキャッシュ情報を追跡し、
        `print_cache_profile` 関数で表示できるようにします。また、NumPy配列をハッシュ可能にするための
        変換と、出力配列を読み取り専用にする処理も内部的に行います。

        Args:
            cache_decorator: キャッシュを行うデコレータ（例: `functools.lru_cache`）。

        Returns:
            関数をラップするデコレータ。このデコレータを適用すると、関数はキャッシュされ、
            そのキャッシュ情報がプロファイリングの対象となり、NumPy配列の引数はハッシュ可能に変換され、
            返り値のNumPy配列は読み取り専用になります。
        """

        def array_to_hashable(f):
            # tuple（hashableなNumpy配列）を受け取る関数をNumpy配列を受け取る関数に変換するデコレータ
            @wraps(f)
            def wrapper(*args, **kwargs):
                new_args = []
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        new_args.append(tuple(map(float, arg)))
                    elif isinstance(arg, tuple):
                        assert False
                    else:
                        new_args.append(arg)
                return f(*new_args, **kwargs)

            return wrapper

        def hashable_to_array(f):
            # Numpy配列を受け取る関数をtuple（hashableなNumpy配列）を受け取る関数に変換する
            @wraps(f)
            def wrapper(*args, **kwargs):
                new_args = []
                for arg in args:
                    if isinstance(arg, tuple):
                        new_args.append(np.array(arg))
                    elif isinstance(arg, np.ndarray):
                        assert False
                    else:
                        new_args.append(arg)
                return f(*new_args, **kwargs)

            return wrapper

        def decorator(f):
            f = hashable_to_array(f)
            f = cached_func = cache_decorator(f)
            self._register_for_cache_profile(cached_func)
            f = array_to_hashable(f)
            f = self.returns_readonly_array(f)
            setattr(f, "cache_info", cached_func.cache_info)
            return f

        return decorator


class FullGateBuilder:
    """
    list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
    関係ないqubitにIdentityを挿入して
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    という(2**n_qubit, 2**n_qubit)行列をつくる.
    """

    def __init__(self, n_qubit):
        self._n_qubit = n_qubit
        self._gates: dict[int, np.ndarray] = {}

    def put(self, i_qubit: int, mat: np.ndarray) -> "FullGateBuilder":
        assert 0 <= i_qubit < self._n_qubit, f"Invalid qubit index: {i_qubit}"
        assert i_qubit not in self._gates, f"Duplicated qubit index: {i_qubit}"
        self._gates[i_qubit] = mat
        return self

    def build(self) -> np.ndarray:
        # 1-qubit gateを並べてnp.kronでreduceする
        single_gates = [
            self._gates.pop(i) if i in self._gates
            else I_mat
            for i in range(self._n_qubit)
        ]
        assert not self._gates, self._gates
        # noinspection PyTypeChecker
        return reduce(np.kron, single_gates)


_mat_id_to_mat_mapper = {
    "I": I_mat,
    "X": X_mat,
    "Y": Y_mat,
    "Z": Z_mat,
}


@cache
def fullgate(n_qubit: int, text: str) -> np.ndarray:
    """
    指定された形式のテキスト文字列から、n_qubit量子ビットの全ゲート（行列）を生成する。

    Args:
        n_qubit (int): 全ゲートが作用する量子ビットの数。
        text (str): ゲートの指定を表す文字列。各ゲートは"MATidQubitIndex"の形式で表され、カンマ区切りで連結される。
            例えば、"Y1,Z2"は、1番目の量子ビットにYゲート、2番目の量子ビットにZゲートを適用することを示す。
            MATidは大文字の'I', 'X', 'Y', 'Z'のいずれかである必要がある。

    Returns:
        np.ndarray: 生成された全ゲートを表す(2**n_qubit, 2**n_qubit)のNumPy配列。

    Raises:
        AssertionError:
            - qubit indexが無効な場合（0 <= i_qubit < n_qubitを満たさない）。
            - 同じqubit indexに対して複数のゲートが指定された場合。
            - qubit indexの順序が昇順でない場合。

    Examples:
        >>> fullgate(n_qubit=3, text="X0,Z2")
        array([[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -1.+0.j  0.+0.j -0.+0.j]
            [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]
            [ 0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -1.+0.j]
            [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j -1.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j]
            [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j -0.+0.j  0.+0.j -1.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j]]
        >>> i, j = 0, 2
        >>> print(fullgate(3, f"X{i},Z{j}"))  # 変数でインデックスを指定したい場合
        array([[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -1.+0.j  0.+0.j -0.+0.j]
            [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]
            [ 0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -1.+0.j]
            [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j -1.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j]
            [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
            [ 0.+0.j -0.+0.j  0.+0.j -1.+0.j  0.+0.j -0.+0.j  0.+0.j -0.+0.j]])
    """

    # fullgate(n_qubit=4, text="Y1,Z2") のような形式で I(0) * Y(1) * Z(2) * I(3) のゲートを作る
    # キャッシュされているので高速
    builder = FullGateBuilder(n_qubit)
    qubit_indexes_ordered = []
    for token in text.split(","):
        mat_id, i_qubit = token[0], int(token[1:])
        qubit_indexes_ordered.append(i_qubit)
        mat = _mat_id_to_mat_mapper[mat_id]
        builder.put(i_qubit, mat)
    assert qubit_indexes_ordered == sorted(qubit_indexes_ordered), \
        f"Qubit indexes are not ordered: {qubit_indexes_ordered}"
    gate = builder.build()
    gate.setflags(write=False)
    return gate


def ham_to_gate_mat(ham: np.ndarray, timestep: float) -> np.ndarray:
    # ハミルトニアンを対角化してゲート行列をつくる
    # H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    datatime_evol_op = np.dot(
        np.dot(
            eigen_vecs,
            np.diag(np.exp(-1j * timestep * diag))
        ),
        eigen_vecs.T.conj(),
    )  # e^-iHT
    return datatime_evol_op


def gate_mat_to_gate(gate_mat: np.ndarray) -> qulacs.QuantumGateMatrix:
    # ゲート行列をqulacsのゲートにする
    assert gate_mat.ndim == 2 and gate_mat.shape[0] == gate_mat.shape[1], gate_mat.shape
    assert math.log2(gate_mat.shape[0]).is_integer(), gate_mat.shape
    # ^ size of gate_mat must be power of 2
    n_qubit = int(math.log2(gate_mat.shape[0]))
    gate = qulacs.gate.DenseMatrix([i for i in range(n_qubit)], gate_mat)
    return gate


def ham_to_gate(ham: np.ndarray, timestep: float) -> qulacs.QuantumGateMatrix:
    # ハミルトニアンをqulacsのゲートにする
    datatime_evol_op = ham_to_gate_mat(ham, timestep)
    gate = gate_mat_to_gate(datatime_evol_op)
    return gate
