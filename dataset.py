import pandas as pd

INPUT = "hour.csv"
OUTPUT = "hour_128kb.csv"
MAX_BYTES = 128 * 1024

df = pd.read_csv(INPUT)

# Keep chronological order
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

# Remove columns that should NOT be used as predictive features
# casual + registered directly make up cnt, so they cause target leakage.
df = df.drop(columns=["instant", "casual", "registered"])

# Keep every 10th observation to preserve the whole 2011–2012 timeline
small = df.iloc[::10].copy()

# Reduce until file is <= 128 KB
while len(small) > 100:
    small.to_csv(OUTPUT, index=False)

    if __import__("os").path.getsize(OUTPUT) <= MAX_BYTES:
        break

    small = small.iloc[:-100]

print("Rows:", len(small))
print("File size:", __import__("os").path.getsize(OUTPUT), "bytes")
print("Saved:", OUTPUT)