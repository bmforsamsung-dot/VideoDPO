import argparse
import copy
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="pair.json or flat prompt/sample json",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="optional metadata.json path for resolving video_path and win/lose type",
    )
    parser.add_argument("--output", type=str, required=True, help="group json path")
    parser.add_argument(
        "--output_metadata_json",
        type=str,
        default=None,
        help="optional compact metadata.json output with re-indexed video_idx",
    )
    parser.add_argument("--losers_per_prompt", type=int, default=5)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    metadata = None
    if args.metadata_json is not None:
        with open(args.metadata_json, "r") as f:
            metadata = json.load(f)

    groups = defaultdict(lambda: {"frame_caption": "", "winner": None, "losers": []})
    # Mode A: existing pair.json format
    # [{"video1":0, "video2":1, "frame_caption":"..."}, ...]
    if len(data) > 0 and "video1" in data[0] and "video2" in data[0]:
        for item in data:
            frame_caption = item.get("frame_caption", "")
            winner_idx = int(item["video1"])
            loser_idx = int(item["video2"])
            pid = f"cap::{frame_caption.strip()}::winner::{winner_idx}"
            groups[pid]["frame_caption"] = frame_caption

            if groups[pid]["winner"] is None:
                winner_obj = {
                    "video_idx": winner_idx,
                    "is_real": True,
                    "sa_score": 1.0,
                    "pc_score": 1.0,
                }
                if metadata is not None:
                    winner_obj["video_path"] = metadata[winner_idx]["basic"]["clip_path"]
                groups[pid]["winner"] = winner_obj

            loser_obj = {
                "video_idx": loser_idx,
                "sa_score": float(item.get("sa_score", 0.0)),
                "pc_score": float(item.get("pc_score", 0.0)),
            }
            if metadata is not None:
                loser_obj["video_path"] = metadata[loser_idx]["basic"]["clip_path"]
            groups[pid]["losers"].append(loser_obj)
    else:
        # Mode B: flat sample list format
        # [{"prompt_id":"...", "video_idx":..., "is_real":bool, ...}, ...]
        for item in data:
            pid = item["prompt_id"]
            groups[pid]["frame_caption"] = item.get("frame_caption", "")
            if item.get("is_real", False):
                winner_obj = {
                    "video_idx": int(item["video_idx"]),
                    "is_real": True,
                    "sa_score": 1.0,
                    "pc_score": 1.0,
                }
                if metadata is not None:
                    widx = int(item["video_idx"])
                    winner_obj["video_path"] = metadata[widx]["basic"]["clip_path"]
                groups[pid]["winner"] = winner_obj
            else:
                loser_obj = {
                    "video_idx": int(item["video_idx"]),
                    "sa_score": float(item.get("sa_score", 0.0)),
                    "pc_score": float(item.get("pc_score", 0.0)),
                }
                if metadata is not None:
                    lidx = int(item["video_idx"])
                    loser_obj["video_path"] = metadata[lidx]["basic"]["clip_path"]
                groups[pid]["losers"].append(loser_obj)

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

    # Optional: build compact metadata.json aligned to group.json indices.
    # This helps when you want a group-only subset dataset.
    if args.output_metadata_json is not None:
        if metadata is None:
            raise ValueError("--output_metadata_json requires --metadata_json")

        used_indices = set()
        for g in out:
            used_indices.add(int(g["winner"]["video_idx"]))
            for l in g["losers"]:
                used_indices.add(int(l["video_idx"]))
        used_indices = sorted(list(used_indices))

        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
        compact_meta = []
        for old_idx in used_indices:
            item = copy.deepcopy(metadata[old_idx])
            if "basic" not in item:
                item["basic"] = {}
            item["basic"]["source_globalidx"] = int(
                item["basic"].get("globalidx", old_idx)
            )
            item["basic"]["globalidx"] = int(old_to_new[old_idx])
            compact_meta.append(item)

        for g in out:
            g["winner"]["video_idx"] = old_to_new[int(g["winner"]["video_idx"])]
            for l in g["losers"]:
                l["video_idx"] = old_to_new[int(l["video_idx"])]

        with open(args.output, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        with open(args.output_metadata_json, "w") as f:
            json.dump(compact_meta, f, ensure_ascii=False, indent=2)
        print(
            f"Saved compact metadata ({len(compact_meta)} videos) -> {args.output_metadata_json}"
        )


if __name__ == "__main__":
    main()
