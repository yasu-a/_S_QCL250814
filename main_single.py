import argparse

from tqdm import tqdm

from generate_random_seed import generate_random_seed
from model.run_param import RunParam
from model.value import FuncType, MethodType, OptMethodType
from run import run


def _main(
        *,
        n_run: int = 100,
        nqubit: int,
        func_type: FuncType,
        method: MethodType,
        opt_method: OptMethodType = "BFGS",
        max_iter: int = 1000,
        abs_tol: float = 1e-7,
        dataset_noise_std: float = 0.01,
) -> None:
    """メイン実行関数
    
    Args:
        n_run: 実行回数
        nqubit: qubit数（必須）
        func_type: 学習する関数の種類（必須）
        method: データアップロードの手法（必須）
        opt_method: 最適化手法
        max_iter: 最適化の最大イテレーション回数
        abs_tol: 最適化でコストがこれ以下になったら中断
    """
    bar = tqdm(range(n_run))
    for _ in bar:
        seed_system = generate_random_seed()
        seed_time_evol = generate_random_seed()
        seed_theta_init = generate_random_seed()

        bar.set_description(
            f"RUN {n_run} TIMES: "
            f"{nqubit=}, {func_type=}, {method=}, {opt_method=}, {max_iter=}, {abs_tol=}, "
            f"seed_system={seed_system}, seed_time_evol={seed_time_evol}, seed_theta_init={seed_theta_init}"
        )

        run_param = RunParam(
            nqubit=nqubit,
            func_type=func_type,
            method=method,
            opt_method=opt_method,
            max_iter=max_iter,
            abs_tol=abs_tol,
            seed_system=seed_system,
            seed_time_evol=seed_time_evol,
            seed_theta_init=seed_theta_init,
            dataset_noise_std=dataset_noise_std,
        )
        run(run_param)


def main() -> None:
    """コマンドライン引数を解析してメイン処理を実行するエントリポイント"""
    parser = argparse.ArgumentParser(description="量子回路学習プログラム")

    # 必須パラメータ
    parser.add_argument(
        "--n-run", "-nr",
        type=int,
        required=True,
        help="実行回数"
    )
    parser.add_argument(
        "--n-qubit", "-nq",
        type=int,
        required=True,
        help="qubit数"
    )
    parser.add_argument(
        "--func-type", "-f",
        choices=["gauss10", "gauss5", "gauss3", "tri"],
        required=True,
        help="学習する関数の種類"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["conv", "pow_3", "nonint"],
        required=True,
        help="データアップロードの手法"
    )

    # オプションパラメータ（デフォルト値付き）
    parser.add_argument(
        "--opt-method",
        choices=["Nelder-Mead", "SLSQP", "BFGS"],
        default="BFGS",
        help="最適化手法（デフォルト: BFGS）"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="最適化の最大イテレーション回数（デフォルト: 1000）"
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-7,
        help="最適化でコストがこれ以下になったら中断（デフォルト: 1e-7）"
    )
    parser.add_argument(
        "--dataset-noise-std", "--noise",
        type=float,
        default=0.01,
        help="データセットに追加するガウシアンノイズの標準偏差",
    )

    args = parser.parse_args()

    # アンダースコアに変換してキーワード引数として渡す
    _main(
        n_run=args.n_run,
        nqubit=args.n_qubit,
        func_type=args.func_type,
        method=args.method,
        opt_method=args.opt_method,
        max_iter=args.max_iter,
        abs_tol=args.abs_tol,
        dataset_noise_std=args.dataset_noise_std,
    )


if __name__ == "__main__":
    main()
