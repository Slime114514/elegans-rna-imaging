import os
import csv
import pandas as pd


# =========================
# Note: the column indices of h and is_signal differ between z and t modes. z=(9,10), t=(5,6)
# =========================
input_folder = r"path/to/your/negative_control_pixels"


# =========================
# Output file name
# =========================
output_file = os.path.join(input_folder, "negative_control_mean_h_summary.csv")


# =========================
# Temporarily store the results for each file
# =========================
results = []

weighted_h_sum = 0.0
total_valid_pixels = 0


# =========================
# Read each CSV file whose name contains "pixels" one by one
# =========================
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".csv"):
        continue

    if "pixels" not in filename:
        continue

    file_path = os.path.join(input_folder, filename)

    print(f"\nProcessing file: {filename}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  [SKIPPED] Failed to read file: {e}")
        continue

    # Note the column indices for the h and is_signal values: z=(9,10), t=(5,6).
    try:
        h_col = df.columns[5]
        is_signal_col = df.columns[6]
    except IndexError:
        print("  [SKIPPED] Not enough columns in the file to read the expected columns")
        continue

    # Convert to numeric to avoid calculation errors caused by strings or empty values
    df[h_col] = pd.to_numeric(df[h_col], errors="coerce")
    df[is_signal_col] = pd.to_numeric(df[is_signal_col], errors="coerce")

    # Keep only rows where is_signal == 1 and h is valid
    valid_df = df[(df[is_signal_col] == 1) & (df[h_col].notna())]

    valid_pixel_count = len(valid_df)

    if valid_pixel_count == 0:
        h_mean = float("nan")
        print("  No valid signal pixels in this file; h mean recorded as NaN")
    else:
        h_mean = valid_df[h_col].mean()

        weighted_h_sum += valid_df[h_col].sum()
        total_valid_pixels += valid_pixel_count

        print(f"  Current file h mean: {h_mean}")
        print(f"  Current file valid pixel count: {valid_pixel_count}")

    results.append({
        "filename": filename,
        "h_mean": h_mean,
        "valid_pixel_count": valid_pixel_count
    })


# =========================
# Compute the overall weighted mean
# =========================
if total_valid_pixels > 0:
    weighted_mean_h = weighted_h_sum / total_valid_pixels
else:
    weighted_mean_h = float("nan")


print("\n==============================")
print("All files processed")
print(f"Total valid pixels: {total_valid_pixels}")
print(f"h weighted mean: {weighted_mean_h}")
print("==============================")


# =========================
# Save the results to the original folder
# =========================
with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow(["filename", "h_mean", "valid_pixel_count"])

    for item in results:
        writer.writerow([
            item["filename"],
            item["h_mean"],
            item["valid_pixel_count"]
        ])

    # Write the weighted mean on the last line
    writer.writerow([])
    writer.writerow(["weighted_mean_h", weighted_mean_h, total_valid_pixels])


print(f"\nResults saved to: {output_file}")