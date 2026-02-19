from pathlib import Path
import pandas as pd

def rewrite_video_paths(
    csv_in,
    csv_out=None,
    old_root="/mnt/sdc1/jiali/data/ego-exo",
    new_root="/mnt/data2/jiali/data/egoexo4d",
    anchor="frame_aligned_videos",
    insert=("downscaled", "448"),
):
    old_root = Path(old_root)
    new_root = Path(new_root)

    df = pd.read_csv(csv_in)
    if "video_path" not in df.columns:
        raise ValueError("CSV must contain a 'video_path' column")

    def map_one(p_str: str) -> str:
        p = Path(str(p_str))
        try:
            rel = p.relative_to(old_root)  # only rewrite if under old_root
        except Exception:
            return str(p)  # leave untouched

        parts = list(rel.parts)
        try:
            i = parts.index(anchor)
        except ValueError:
            # No anchor found: just swap the root
            return str(new_root.joinpath(*parts))

        new_parts = parts[: i + 1] + list(insert) + parts[i + 1 :]
        return str(new_root.joinpath(*new_parts))

    before = df["video_path"].astype(str)
    df["video_path"] = before.map(map_one)
    changed = (df["video_path"] != before).sum()

    out = csv_out or csv_in
    df.to_csv(out, index=False)
    print(f"Rewrote {changed} paths → {out}")
    return out

# Example:
rewrite_video_paths("ood_group_by_take/split_assignments.csv", "ood_group_by_take/split_assignments_downscaled.csv")
