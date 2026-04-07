import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_json", type=str, required=True)
    parser.add_argument("--metadata_json", type=str, required=True)
    parser.add_argument("--expected_losers", type=int, default=5)
    args = parser.parse_args()

    with open(args.group_json, "r") as f:
        groups = json.load(f)
    with open(args.metadata_json, "r") as f:
        meta = json.load(f)

    n = len(meta)
    errors = 0
    for i, g in enumerate(groups):
        widx = int(g["winner"]["video_idx"])
        if not (0 <= widx < n):
            print(f"[{i}] winner idx out of range: {widx}")
            errors += 1
        losers = g.get("losers", [])
        if len(losers) != args.expected_losers:
            print(f"[{i}] losers != {args.expected_losers}: {len(losers)}")
        for l in losers:
            vidx = int(l["video_idx"])
            if not (0 <= vidx < n):
                print(f"[{i}] loser idx out of range: {vidx}")
                errors += 1
            for k in ["sa_score", "pc_score"]:
                s = float(l.get(k, -1))
                if s < 0.0 or s > 1.0:
                    print(f"[{i}] {k} out of [0,1]: {s}")
                    errors += 1

    if errors == 0:
        print("Validation passed.")
    else:
        print(f"Validation failed with {errors} errors.")


if __name__ == "__main__":
    main()
