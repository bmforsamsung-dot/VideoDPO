import argparse
import copy
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_json",
        type=str,
        required=True,
        help="metadata.json path for resolving video_path and win/lose type",
    )
    parser.add_argument(
        "--winner_path_keyword",
        type=str,
        default="winvideos",
        help="keyword in clip_path used to identify winner(real) videos in metadata-only mode",
    )
    parser.add_argument(
        "--loser_path_keyword",
        type=str,
        default="losevideos",
        help="keyword in clip_path used to identify loser videos in metadata-only mode",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="caption",
        choices=["caption", "prompt_id"],
        help="group key in metadata-only mode",
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

    metadata = None
    if args.metadata_json is not None:
        with open(args.metadata_json, "r") as f:
            metadata = json.load(f)
    if metadata is None:
        raise ValueError("Provide --metadata_json")

    groups = defaultdict(lambda: {"frame_caption": "", "winner": None, "losers": []})
    # Build groups directly from metadata.json.
    # Winner/loser are inferred from clip_path keywords.
    by_caption = defaultdict(list)
    for idx, item in enumerate(metadata):
        if args.group_by == "prompt_id":
            key = item.get("misc", {}).get("prompt_id", "")
        else:
            key = item.get("misc", {}).get("frame_caption", [""])
            key = key[0] if isinstance(key, list) else key
        by_caption[key].append((idx, item))

    for cap, samples in by_caption.items():
        winners = []
        losers = []
        for idx, item in samples:
            p = str(item.get("basic", {}).get("clip_path", "")).lower()
            entry = {
                "video_idx": int(idx),
                "sa_score": 0.0,
                "pc_score": 0.0,
                "video_path": item.get("basic", {}).get("clip_path", ""),
            }
            if args.winner_path_keyword.lower() in p:
                winners.append(entry)
            elif args.loser_path_keyword.lower() in p:
                losers.append(entry)
            else:
                # ignore unknown tag path in metadata-only mode
                continue
        if len(winners) == 0 or len(losers) == 0:
            continue
        winner_obj = {
            "video_idx": winners[0]["video_idx"],
            "is_real": True,
            "sa_score": 1.0,
            "pc_score": 1.0,
            "video_path": winners[0]["video_path"],
        }
        pid = f"cap::{cap.strip()}::winner::{winner_obj['video_idx']}"
        groups[pid]["frame_caption"] = cap
        groups[pid]["winner"] = winner_obj
        groups[pid]["losers"] = losers[: args.losers_per_prompt]

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
