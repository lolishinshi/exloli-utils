import sqlite3
import json
from typing import List, Tuple, Generator
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

plt.rcParams["font.sans-serif"] = ["SimSun"]

CONN = sqlite3.connect("./db.sqlite")
C = CONN.cursor()


def translate(string: str, __d={}) -> str:
    if __d == {}:
        for data in json.load(open("./db.text.json"))["data"]:
            if data["namespace"] == "artist":
                __d.update(data["data"])
    return __d.get(string.split(" | ")[0], {}).get("name", string)


def escape(string: str) -> str:
    return string.split(" | ")[0].replace(" ", "_")


# https://www.jianshu.com/p/4d2b45918958
def wilson_score_norm(mean, var, total, p_z=2.0):
    # 均值方差需要归一化，以符合正态分布的分位数
    score = (
        mean
        + (np.square(p_z) / (2.0 * total))
        - ((p_z / (2.0 * total)) * np.sqrt(4.0 * total * var + np.square(p_z)))
    ) / (1 + np.square(p_z) / total)
    return score


def calc_score(votes, p_z=1.959):
    arr = np.array([0, 0, 0, 0, 0])
    for i in votes:
        arr += np.array(i)
    min, max = 1.0, 5.0
    arr = sum(([i + 1] * arr[i] for i in range(5)), [])

    values = np.array(arr)
    norm_values = (values - min) / (max - min)
    total = norm_values.size
    mean = np.mean(norm_values)
    var = np.var(norm_values)

    return wilson_score_norm(mean, var, total, p_z) * 100


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
    artists = defaultdict(lambda: [])
    # TODO: 过滤 poll_id 相同的本子
    for row in C.execute("SELECT message_id, votes, tags FROM gallery WHERE score > 0"):
        mid, votes, tags = row
        if tags == "":
            print(f"Error: {mid}")
            continue

        for artist in get_artist(json.loads(tags)):
            artists[artist].append(json.loads(votes))

    # x - 作品数量，y - 平均分
    data = []
    for artist, votes in artists.items():
        data.append(
            {
                "score": calc_score(votes),
                "count": sum(sum(votes, [])),
                "name": translate(artist)
            }
        )

    max_score = max(data, key=lambda x: x["score"])["score"]
    max_count = max(data, key=lambda x: x["count"])["count"]
    # 综合排名：-max(x["score"] / max_score, x["count"] / max_count)
    data.sort(key=lambda x: -x["score"])

    data = [d for d in data if d["count"] >= 0][:20]

    x = [d["score"] for d in data]
    y = [d["count"] for d in data]
    labels = [d["name"] for d in data]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    texts = []
    for _x, _y, label in zip(x, y, labels):
        texts.append(plt.text(_x, _y, label, fontsize=12))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='b'))

    plt.xlabel('平均分')
    plt.ylabel('投票人数')
    plt.show()


if __name__ == "__main__":
    main()
    CONN.commit()
    CONN.close()
