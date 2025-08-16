import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from model.run_record import GlobalRecord
from model.value import FuncType, MethodType, OptMethodType

BASE_DIR_PATH = os.path.join("./", "sato_tougou")
os.makedirs(BASE_DIR_PATH, exist_ok=True)
print(f"{ BASE_DIR_PATH=}")


@dataclass(frozen=True)
class RecordKey:
    nqubit: int  # qubit数
    func_type: FuncType  # 学習する関数の種類（ガウス分布，三角波）
    method: MethodType  # データアップロードの手法（従来手法＝均一回転角，3のi乗，非可積分系）
    opt_method: OptMethodType  # 最適化手法

    @classmethod
    def from_record(cls, record: GlobalRecord):
        return RecordKey(
            nqubit=record.nqubit,
            func_type=record.func_type,
            method=record.method,
            opt_method=record.opt_method,
        )


FOLDER_NAME_KEYS: list[Callable[[RecordKey], str]] = [
    lambda x: x.opt_method,
    lambda x: f"{x.method}-{x.func_type}",
    lambda x: f"{x.nqubit}-qubit"
]

FILENAME_KEY: Callable[[GlobalRecord], str] \
    = lambda x: f"{x.seed_system}-{x.seed_time_evol}-{x.seed_theta_init}.json"


def _create_folder_path(record: RecordKey) -> str:
    path_parts = [key(record) for key in FOLDER_NAME_KEYS]
    return os.path.join(BASE_DIR_PATH, *path_parts)


def _create_filepath(record: GlobalRecord) -> str:
    filepath = os.path.join(_create_folder_path(RecordKey.from_record(record)),
                            FILENAME_KEY(record))
    return filepath


def count_records(record: RecordKey) -> int:
    path = _create_folder_path(record)
    if not os.path.exists(path):
        return 0
    return len(os.listdir(path))


def put_record(record: GlobalRecord):
    filepath = _create_filepath(record)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(record.to_json(), f, indent=2, ensure_ascii=True)
    print(f"Result saved as '{filepath!s}'")


def iter_json_path() -> Iterable[str]:
    yield from map(str, Path(BASE_DIR_PATH).rglob("*.json"))
