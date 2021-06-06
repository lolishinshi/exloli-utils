import sqlite3
import json
from typing import List, Tuple, Generator
from collections import defaultdict
import numpy as np

CONN = sqlite3.connect("./db.sqlite")
C = CONN.cursor()


def translate(string: str, __d={}):
    if __d == {}:
        for data in json.load(open("../db.text.json"))["data"]:
            if data["namespace"] == "artist":
                __d = data["data"]
    return __d.get(string.split(" | ")[0], {}).get("name", string)


def escape(string: str) -> str:
    return string.split(" | ")[0].replace(" ", "_")


# https://www.jianshu.com/p/4d2b45918958
def wilson_score_norm(mean, var, total, p_z=2.):
    # 均值方差需要归一化，以符合正态分布的分位数
    score = (mean + (np.square(p_z) / (2. * total))
             - ((p_z / (2. * total)) * np.sqrt(4. * total * var + np.square(p_z)))) / \
            (1 + np.square(p_z) / total)
    return score

def calc_score(arr, p_z=1.281):
    max = 5.  # 五星评价的最大值
    min = 1.  # 五星评价的最小值
    arr = sum(([i + 1] * arr[i] for i in range(5)), [])
    #if len(arr) >= 10:
    #    del arr[0], arr[-1]
    values = np.array(arr)  # 示例

    norm_values = (values - min) / (max - min)  # 归一化
    total = norm_values.size  # 总数
    mean = np.mean(norm_values)  # 归一化后的均值
    var = np.var(norm_values)  # 归一化后的方差
    return wilson_score_norm(mean, var, total, p_z)


def get_artist(tags: List[Tuple[str, List[str]]]) -> Generator[str, None, None]:
    for cat, tag in tags:
        if cat == "artist":
            yield from tag
            return
    for cat, tag in tags:
        if cat == "group":
            yield from tag
    return None


def main():
    artists = defaultdict(lambda: np.array([0, 0, 0, 0, 0]))
    for row in C.execute("SELECT message_id, votes, tags FROM gallery WHERE score > 0"):
        mid, votes, tags = row
        if tags == "":
            print(f"Error: {mid}")
            continue

        for artist in get_artist(json.loads(tags)):
            artists[artist] += np.array(json.loads(votes))

    artists = [[artist, calc_score(score)] for artist, score in artists.items()]
    artists.sort(key=lambda x: -x[1])
    for idx, (artist, score) in enumerate(artists[:20]):
        print(
            f"`{idx+1:<2}  {score * 100:<6.1f}`  #{translate(artist)} #{escape(artist)}"
        )


if __name__ == "__main__":
    main()
    CONN.commit()
    CONN.close()
