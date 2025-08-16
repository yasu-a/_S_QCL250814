import collections
import datetime
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from tqdm import tqdm

import repo
from model.run_record import GlobalRecord
from model.value import OptMethodType, FuncType, MethodType

OUTPUT_DIR_PATH = "out"
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)


def read_data():
    records: list[GlobalRecord] = []
    mtimes: list[datetime.datetime] = []
    paths: list[str] = []

    def read_file(path: str) -> tuple[str, int, str]:
        with open(path, "r") as f:
            return f.read(), os.stat(path).st_mtime, path

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(read_file, filepath)
            for filepath in repo.iter_json_path()
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            body, mtime, path = future.result()
            records.append(GlobalRecord.from_json(json.loads(body)))
            mtimes.append(datetime.datetime.fromtimestamp(mtime))
            paths.append(path)

    return records, mtimes, paths


def check_seed_duplication(records: list[GlobalRecord], paths: list[str]):
    @dataclass(frozen=True)
    class SeedConfig:
        seed_train: int
        seed_system: int
        seed_time_evol: int
        seed_theta_init: int

        @classmethod
        def from_record(cls, r: GlobalRecord):
            return cls(
                seed_train=r.seed_train,
                seed_system=r.seed_system,
                seed_time_evol=r.seed_time_evol,
                seed_theta_init=r.seed_theta_init,
            )

    seed_config_to_records = collections.defaultdict(list)
    for i in range(len(records)):
        seed_config = SeedConfig.from_record(records[i])
        seed_config_to_records[seed_config].append((records[i], paths[i]))

    n_duplicated = 0
    for seed_config, items in seed_config_to_records.items():
        if len(items) <= 1:
            continue
        print(f"*** DUPLICATED SEED CONFIG: {seed_config}")
        for record, path in items:
            print(f" - {path}")
        print()
        n_duplicated += 1

    assert n_duplicated == 0, f"{n_duplicated} DUPLICATED ENTRIES EXISTS"


def dump_as_csv(records, mtimes):
    df_records = pd.DataFrame([
        {
            "opt_method": r.opt_method,
            "nqubit": r.nqubit,
            "c_depth": r.c_depth,
            "func_type": r.func_type,
            "method": r.method,
            "seed_train": r.seed_train,
            "seed_system": r.seed_system,
            "seed_time_evol": r.seed_time_evol,
            "seed_theta_init": r.seed_theta_init,
            "max_n_iter": r.iteration_records[-1].n_iter,
            "opt_maxiter": r.opt_maxiter,
            "train_loss_opt": r.train_loss_opt,
            "mtime": mtime,
            "record": r,
        } for r, mtime in zip(records, mtimes)
    ])
    df_records.iloc[:, :-
    1].to_csv(os.path.join(OUTPUT_DIR_PATH, "data.csv"), index=False)

    df_num_data = df_records.groupby(
        ["opt_maxiter", "opt_method", "func_type", "method", "nqubit", "opt_maxiter"])[
        "record"].count()
    df_num_data.to_csv(os.path.join(OUTPUT_DIR_PATH, "num_data.csv"))

    df_n_iter = (
        df_records
        .groupby(["opt_maxiter", "opt_method", "func_type", "method", "nqubit"])["max_n_iter"]
        .describe()[["count", "mean", "min", "25%", "50%", "75%", "max"]]
        .astype(int)
    )
    df_n_iter.to_csv(os.path.join(OUTPUT_DIR_PATH, "n_iter.csv"))

    df_not_converged = df_records.groupby(
        ["opt_maxiter", "opt_method", "func_type", "method", "nqubit"]).apply(
        lambda g: pd.Series({
            "Count": len(g),
            "# of convergence condition unsatisfied": (g.max_n_iter == g.opt_maxiter).sum(),
            "Ratio [%]": (g.max_n_iter == g.opt_maxiter).sum() / len(g) * 100,
        }).astype(int)
    )
    df_not_converged.to_csv(os.path.join(OUTPUT_DIR_PATH, "not_converged.csv"))


def record_summary(r: GlobalRecord) -> str:
    return f"{r.method} - {r.func_type}, nqubit={r.nqubit}, {r.opt_method}, {r.train_loss_opt:.2e}, n_iter={max(it.n_iter for it in r.iteration_records)}"


def iter_filtered_records(
        records,
        *,
        opt_method: OptMethodType = None,
        nqubit: int = None,
        func_type: FuncType = None,
        method: MethodType = None,
) -> Iterable[GlobalRecord]:
    for r in records:
        if opt_method is not None and r.opt_method != opt_method:
            assert isinstance(opt_method,
                              str), f"iter_filtered_records: invalid opt_method={opt_method!r}"
            continue
        if nqubit is not None and r.nqubit != nqubit:
            assert isinstance(
                nqubit, int), f"iter_filtered_records: invalid nqubit={nqubit!r}"
            continue
        if func_type is not None and r.func_type != func_type:
            assert isinstance(func_type,
                              str), f"iter_filtered_records: invalid func_type={func_type!r}"
            continue
        if method is not None and r.method != method:
            assert isinstance(
                method, str), f"iter_filtered_records: invalid method={method!r}"
            continue
        yield r


def save_one_data(
        records: list[GlobalRecord],
        *,
        filename_prefix: str,
        opt_method: OptMethodType,
        n_qubit: int,
        func_type: FuncType,
        method: MethodType,
        sample_index: int,
        show=False,
) -> None:
    try:
        print(
            dict(
                opt_method=opt_method,
                nqubit=n_qubit,
                func_type=func_type,
                method=method,
            )
        )
        active_record = list(
            iter_filtered_records(
                records,
                opt_method=opt_method,
                nqubit=n_qubit,
                func_type=func_type,
                method=method,
            )
        )[sample_index]
    except IndexError:
        print("データがありません")
    else:
        # きれいに表示
        with open(
                os.path.join(
                    OUTPUT_DIR_PATH,
                    f"{filename_prefix}_{opt_method}_{n_qubit}_{func_type}_{method}.txt",
                ),
                "w",
        ) as f_out:
            print(
                dict(
                    opt_method=opt_method,
                    nqubit=n_qubit,
                    func_type=func_type,
                    method=method,
                ),
                file=f_out,
            )

            for f in fields(active_record):
                v = getattr(active_record, f.name)
                if f.name == "iteration_records":
                    v_str = f"<list of iteration record: cost_init={v[0].train_loss:.5f}, cost_opt={v[-1].train_loss:.5f}, n_iter={v[-1].n_iter}, elasped_time={v[-1].elapsed_time:.0f}sec>"
                elif isinstance(v, np.ndarray):
                    with np.printoptions(suppress=True, precision=5, linewidth=200):
                        v_str = np.array2string(v, separator=", ", sign=" ")
                else:
                    v_str = str(v)

                for lno, line in enumerate(v_str.split("\n")):
                    print((f.name if lno == 0 else '').ljust(20) + " " + line, file=f_out)

        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(active_record.x_train,
                 active_record.y_train, "o", label='Teacher')
        plt.plot(active_record.x_plot_init, active_record.y_plot_init, '--',
                 label='Initial Model Prediction', c='gray')
        plt.plot(active_record.x_plot_opt, active_record.y_plot_opt,
                 label='Final Model Prediction')
        plt.title(record_summary(active_record))
        plt.ylim(-1.1, 1.1)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR_PATH,
                f"{filename_prefix}_{opt_method}_{n_qubit}_{func_type}_{method}.pdf",
            )
        )
        if show:
            plt.show()
        plt.close()


def save_one_data_with_median_cost(
        records: list[GlobalRecord],
        *,
        opt_method: OptMethodType,
        n_qubit: int,
        func_type: FuncType,
        method: MethodType,
) -> None:
    rs = list(
        iter_filtered_records(
            records,
            opt_method=opt_method,
            nqubit=n_qubit,
            func_type=func_type,
            method=method,
        )
    )
    if len(rs) == 0:
        warnings.warn(
            f"{opt_method=!r}, {n_qubit=!r}, {func_type=!r}, {method=!r}のデータがありません"
        )
        return

    cost_array = np.array([r.train_loss_opt for r in rs])
    median_cost = np.median(cost_array)
    median_cost_index = np.argmin(np.abs(cost_array - median_cost))
    median_cost_record = rs[median_cost_index]
    save_one_data(
        [median_cost_record],
        filename_prefix="median_cost",
        opt_method=opt_method,
        n_qubit=n_qubit,
        func_type=func_type,
        method=method,
        sample_index=0,
        show=False,
    )


def show_cost_distribution(
        records: list[GlobalRecord],
        *,
        filename: str,
        target_opt_method: OptMethodType,
        target_nqubit_lst: list[int],
        target_func_type: FuncType,
        target_method_lst: list[MethodType],
        n_samples: int,
        wide: bool = False,
):
    method_text_mapper = {
        "conv": "Encoding with\nuniform rotation angles",
        "pow_3": "Encoding with\nexponential function",
        "nonint": "Encoding with\nnon-integrable system",
    }

    # 画像のサイズ
    if wide:
        plt.figure(figsize=(10 / 3 * len(target_nqubit_lst), 5), dpi=200)
    else:
        plt.figure(figsize=(7 / 3 * len(target_nqubit_lst), 5), dpi=200)

    x_tick_pos = []
    x_tick_labels_1 = []
    x_tick_labels_2 = []
    x_tick_group_id = []
    x = 0
    group_id = 0
    for method in target_method_lst:
        for nqubit in target_nqubit_lst:
            # (method, nqubit) のデータを取り出す
            rs = list(
                iter_filtered_records(
                    records,
                    opt_method=target_opt_method,
                    nqubit=nqubit,
                    func_type=target_func_type,
                    method=method,
                )
            )
            if n_samples is None:
                warnings.warn(
                    f"{n_samples=}が指定されていないため{target_func_type=}, {method=}, {nqubit=}のデータ数は保証されません"
                )
            else:
                rs = rs[:n_samples]
                assert len(
                    rs) == n_samples, f"{target_func_type=!r} {method=!r}, {nqubit=!r}のデータ数{len(rs)}が指定された{n_samples=}に対して不足しています"

            x_tick_pos.append(x)
            x_tick_labels_1.append(f"{nqubit}-qubit")
            x_tick_labels_2.append(f"\n\n{method_text_mapper[method]}")
            x_tick_group_id.append(group_id)

            if len(rs) != 0:
                cost_init = np.array([r.train_loss_init for r in rs])
                cost_opt = np.array([r.train_loss_opt for r in rs])

                # (method, nqubit) の箱ひげ図を作る
                boxplot_options = dict(
                    showfliers=False,
                    whis=1e+99,
                    widths=0.1,
                )
                plt.boxplot(
                    cost_init,
                    **boxplot_options,
                    label="Initial cost",
                    positions=[x - 0.15],
                    boxprops=dict(color='tab:blue'),
                    whiskerprops=dict(color='tab:blue'),
                    capprops=dict(color='tab:blue'),
                    medianprops=dict(color='tab:blue'),

                )
                plt.boxplot(
                    cost_opt,
                    **boxplot_options,
                    label="Optimized cost",
                    positions=[x - 0.05],
                    boxprops=dict(color='tab:red'),
                    whiskerprops=dict(color='tab:red'),
                    capprops=dict(color='tab:red'),
                    medianprops=dict(color='tab:red'),
                )

                cost_init = np.array([r.test_loss_init for r in rs])
                cost_opt = np.array([r.test_loss_opt for r in rs])

                # (method, nqubit) の箱ひげ図を作る
                boxplot_options = dict(
                    showfliers=False,
                    whis=1e+99,
                    widths=0.1,
                )
                plt.boxplot(
                    cost_init,
                    **boxplot_options,
                    label="Initial cost",
                    positions=[x + 0.05],
                    boxprops=dict(color='tab:blue'),
                    whiskerprops=dict(color='tab:blue'),
                    capprops=dict(color='tab:blue'),
                    medianprops=dict(color='tab:blue'),

                )
                plt.boxplot(
                    cost_opt,
                    **boxplot_options,
                    label="Optimized cost",
                    positions=[x + 0.15],
                    boxprops=dict(color='tab:red'),
                    whiskerprops=dict(color='tab:red'),
                    capprops=dict(color='tab:red'),
                    medianprops=dict(color='tab:red'),
                )

            x += 1
        group_id += 1

    x_tick_pos = np.array(x_tick_pos)
    x_tick_labels_1 = np.array(x_tick_labels_1)
    x_tick_labels_2 = np.array(x_tick_labels_2)
    x_tick_group_id = np.array(x_tick_group_id)

    # # 折れ線
    # for g in np.sort(np.unique(x_tick_group_id)):
    #     plt.plot(
    #         x_tick_pos[x_tick_group_id == g] - 0.1,
    #         y_median_init[x_tick_group_id == g],
    #         c="tab:blue",
    #         alpha=0.5,
    #         lw=1,
    #     )
    #     plt.plot(
    #         x_tick_pos[x_tick_group_id == g] + 0.1,
    #         y_median_opt[x_tick_group_id == g],
    #         c="tab:red",
    #         alpha=0.5,
    #         lw=1,
    #     )

    # グラフ内の垂直の仕切り
    for g in np.sort(np.unique(x_tick_group_id))[:-1]:
        plt.axvline(x_tick_pos[x_tick_group_id ==
                               g].max() + 0.5, c="black", lw=1)

    # X軸の設定
    plt.xlim(np.min(x_tick_pos) - 0.5, np.max(x_tick_pos) + 0.5)
    plt.xticks(range(len(x_tick_labels_1)), x_tick_labels_1)

    # x-axis multilevel ticks
    # https://matplotlib.org/stable/gallery/ticks/multilevel_ticks.html
    #  - label the classes:
    sec = plt.gca().secondary_xaxis(location=0)
    sec.set_xticks(
        [x_tick_pos[x_tick_group_id == g].mean()
         for g in np.unique(x_tick_group_id)],
        labels=[x_tick_labels_2[x_tick_group_id == g][0]
                for g in np.unique(x_tick_group_id)],
    )
    sec.tick_params('x', length=0)
    #  - lines between the classes:
    sec2 = plt.gca().secondary_xaxis(location=0)
    sec2.set_xticks(
        [x_tick_pos[x_tick_group_id == g].min() - 0.5 for g in np.unique(x_tick_group_id)] + [
            x_tick_pos[x_tick_group_id == x_tick_group_id.max()].max() + 0.5],
        labels=[],
    )
    sec2.tick_params('x', length=50, width=1)
    plt.gca().set_xlim(x_tick_pos.min() - 0.5, x_tick_pos.max() + 0.5)

    # Y軸の設定
    plt.yscale("log")
    plt.ylabel("Cost")
    plt.grid(axis="y", which="major", color="black", alpha=0.5)
    # plt.grid(axis="y", which="minor", color="black", alpha=0.2)
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=999))
    # plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(0.1, 1, 0.1), numticks=999))
    plt.ylim(1e-8, 1e+1)

    # 重複した凡例を削除
    # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower left")

    # 表示
    print(f" ↓ {target_func_type=}, n={n_samples} for each box")
    if n_samples is None:
        plt.text(0, 1, "NOT AVAILABLE FOR PUBLICATION", size=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PATH, filename))
    plt.show()
    plt.close()


def main():
    records, mtimes, paths = read_data()
    check_seed_duplication(records, paths)
    dump_as_csv(records, mtimes)

    """
    def save_one_data_with_median_cost(
            records: list[GlobalRecord],
            *,
            opt_method: OptMethodType,
            n_qubit: int,
            func_type: FuncType,
            metho
    """

    for opt_method in ["BFGS"]:
        for n_qubit in [2, 3, 4]:
            for func_type in ["gauss5", "tri"]:
                for method in ["conv", "pow_3", "nonint"]:
                    save_one_data_with_median_cost(
                        records,
                        opt_method=opt_method,
                        n_qubit=n_qubit,
                        func_type=func_type,
                        method=method,
                    )

    target_nqubit_lst = [2, 3, 4]
    for func_type in ["gauss5", "tri"]:
        show_cost_distribution(
            records,
            filename=f"{''.join(map(str, target_nqubit_lst))}-qubit_{func_type}.pdf",
            target_opt_method="BFGS",
            target_nqubit_lst=target_nqubit_lst,
            target_func_type=func_type,
            target_method_lst=["conv", "pow_3", "nonint"],
            n_samples=None,
        )


if __name__ == '__main__':
    main()
