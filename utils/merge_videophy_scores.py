import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_json", type=str, required=True)
    parser.add_argument("--sa_csv", type=str, required=True)
    parser.add_argument("--pc_csv", type=str, required=True)
    parser.add_argument("--video_col", type=str, default="videopath")
    parser.add_argument("--video_idx_col", type=str, default=None)
    parser.add_argument("--score_col", type=str, default="score")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--normalize_divisor", type=float, default=5.0)
    args = parser.parse_args()

    with open(args.group_json, "r") as f:
        groups = json.load(f)
    sa_df = pd.read_csv(args.sa_csv)
    pc_df = pd.read_csv(args.pc_csv)

    sa_map = dict(zip(sa_df[args.video_col], sa_df[args.score_col]))
    pc_map = dict(zip(pc_df[args.video_col], pc_df[args.score_col]))
    sa_idx_map = {}
    pc_idx_map = {}
    if args.video_idx_col is not None:
        sa_idx_map = dict(zip(sa_df[args.video_idx_col], sa_df[args.score_col]))
        pc_idx_map = dict(zip(pc_df[args.video_idx_col], pc_df[args.score_col]))

    missing = 0
    for g in groups:
        for loser in g.get("losers", []):
            key = loser.get("video_path", None)
            sa = sa_map.get(key, None) if key is not None else None
            pc = pc_map.get(key, None) if key is not None else None
            if (sa is None or pc is None) and args.video_idx_col is not None:
                vidx = loser.get("video_idx", None)
                sa = sa_idx_map.get(vidx, sa)
                pc = pc_idx_map.get(vidx, pc)
            if sa is None or pc is None:
                missing += 1
                continue
            loser["sa_score"] = float(sa) / args.normalize_divisor
            loser["pc_score"] = float(pc) / args.normalize_divisor

    with open(args.output, "w") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    print(f"Saved merged group json to {args.output}, missing={missing}")


if __name__ == "__main__":
    main()
