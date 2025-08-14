import os


def generate_random_seed() -> int:
    # OSの乱数生成器を使う（os.urandomを利用）
    return int.from_bytes(os.urandom(4), "big")


if __name__ == '__main__':
    def main():
        # 衝突しないことの確認用
        v = []
        from tqdm import tqdm
        for _ in tqdm(range(3000000)):
            v.append(generate_random_seed())
        p = len(set(v)) / len(v)
        print(f"{len(v)=}, {len(set(v))=}, {p=}")  # 誕生日のパラドクスによると0.000116の確率で衝突する 0.999884くらいが妥当


    main()
