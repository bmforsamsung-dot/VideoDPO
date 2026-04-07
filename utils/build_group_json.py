import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="flat pair/list json")
    parser.add_argument("--output", type=str, required=True, help="group json path")
    parser.add_argument("--losers_per_prompt", type=int, default=5)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    groups = defaultdict(lambda: {"frame_caption": "", "winner": None, "losers": []})
    for item in data:
        pid = item["prompt_id"]
        groups[pid]["frame_caption"] = item.get("frame_caption", "")
        if item.get("is_real", False):
            groups[pid]["winner"] = {
                "video_idx": int(item["video_idx"]),
                "is_real": True,
                "sa_score": 1.0,
                "pc_score": 1.0,
            }
        else:
            groups[pid]["losers"].append(
                {
                    "video_idx": int(item["video_idx"]),
                    "sa_score": float(item.get("sa_score", 0.0)),
                    "pc_score": float(item.get("pc_score", 0.0)),
                }
            )

    out = []
    for pid, g in groups.items():
        if g["winner"] is None or len(g["losers"]) == 0:
            continue
        g["losers"] = g["losers"][: args.losers_per_prompt]
        out.append(
            {
                "prompt_id": pid,
                "frame_caption": g["frame_caption"],
                "winner": g["winner"],
                "losers": g["losers"],
            }
        )

    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out)} groups -> {args.output}")


if __name__ == "__main__":
    main()
