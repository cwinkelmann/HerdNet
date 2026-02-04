import pandas as pd

# Images to filter (without extension)
target_images = [
    "01802f75da35434ab373569fffc1fd65a3417aef",
    "018f5ab5b7516a47ff2ac48a9fc08353b533c30f",
    "02033bf9b6c41f5815072434f8d61707cc8ea1fb",
    "02e306916552df0dfe01fa352590ebb5f2a8b8ab",
    "045f3b931fae913307c1f11512e79ddd891cb3ad",
    "04bda8c273b5ef6b29e1f318a1da3e7506a5e4d8",
    "04e8092d743bef891386f3e0ce82155f12aa4035",
]

# Read CSV
df = pd.read_csv("/home/christian/hnee/HerdNet/notebooks/data/test.csv")

# Filter rows where image name (without extension) matches
df["image_stem"] = df["images"].str.rsplit(".", n=1).str[0]
filtered = df[df["image_stem"].isin(target_images)].drop(columns=["image_stem"])

# Save to new file
filtered.to_csv("filtered_annotations.csv", index=False)
print(f"Copied {len(filtered)} annotations for {filtered['images'].nunique()} images")