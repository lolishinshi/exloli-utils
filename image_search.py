# code by @exlolicon
import sys
import pickle
import typing as t
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


DB_PATH = Path(__file__).parent / "db.pickle"
if DB_PATH.exists():
    DB = pickle.load(DB_PATH.open("rb"))
else:
    DB = {}


def orb_detect_and_compute_by_part(orb: cv2.ORB, img: np.ndarray, part: t.Tuple[int, int] = (5, 5)):
    height, width = img.shape
    dh, dw = height // part[0] + 1, width // part[1] + 1

    final_des = []

    for h in range(0, height, dh):
        mh = min(h + dh, height)
        for w in range(0, width, dw):
            mw = min(w + dw, width)
            crop = img[h:mh, w:mw]
            _, des = orb.detectAndCompute(crop, None)
            if des is not None:
                final_des.extend(des)

    return np.array(final_des, dtype="uint8")


def adjust_image_size(img: np.ndarray, shape: t.Tuple[int, int] = (1080, 1920)) -> np.ndarray:
    scale = min(shape[0] / img.shape[0], shape[1] / img.shape[1])
    dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def index_images(path: Path):
    orb: cv2.ORB = cv2.ORB_create(500 // (5 * 5))
    for file in path.rglob('**/*'):
        if file.suffix.lower() not in ['.jpg', '.png']:
            continue
        print(f"Indexing {file}")
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        img = adjust_image_size(img)
        des = orb_detect_and_compute_by_part(orb, img)
        for i in des:
            DB[i.data.tobytes()] = str(file)
    pickle.dump(DB, DB_PATH.open("wb"))


def search_image(path: Path):
    FLANN_INDEX_LSH = 6
    index_params = {
        "algorithm": FLANN_INDEX_LSH,
        "table_number": 6,
        "key_size": 12,
        "multi_probe_level": 1,
    }
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    orb: cv2.ORB = cv2.ORB_create()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _, des = orb.detectAndCompute(img, None)

    files = defaultdict(lambda: 0)
    keys = np.array([np.frombuffer(b, dtype='uint8') for b in DB.keys()])
    matches = flann.knnMatch(des, keys, k=2)
    for match in matches:
        if len(match) != 2:
            continue
        m, n = match
        if m.distance < 0.7 * n.distance:
            files[DB[keys[m.trainIdx].data.tobytes()]] += 1

    for k, v in files.items():
        print(v, k)


def main():
    if len(sys.argv) != 2:
        print(
            "用法：\n"
            f"  索引目录：{sys.argv[0]} <DIR>\n"
            f"  查找图片：{sys.argv[0]} <FILE>\n"
        )
        sys.exit()
    path = Path(sys.argv[1])
    if path.is_dir():
        index_images(path)
    elif path.is_file():
        search_image(path)


if __name__ == '__main__':
    main()
