import os
import pandas as pd


# =========================
# TO MODIFY: input folder path
# =========================
input_folder = r"path/to/your/input_folder"


# =========================
# Enter the negative control mean brightness value at runtime
# =========================
while True:
    try:
        negative_control_h = float(input("Please enter the mean brightness value of the negative control group: "))
        print(f"Using preset negative_control_h = {negative_control_h}")
        break
    except ValueError:
        print("Invalid input, please enter a number.")


# =========================
# Create the output folder
# For example: /path/to/your/experiment
# Output: /path/to/your/experiment_h_net
# =========================
parent_dir = os.path.dirname(input_folder)
folder_name = os.path.basename(os.path.normpath(input_folder))
output_folder = os.path.join(parent_dir, folder_name + "_h_net")

os.makedirs(output_folder, exist_ok=True)

print(f"\nInput folder: {input_folder}")
print(f"Output folder: {output_folder}")
print(f"Negative control group mean brightness value: {negative_control_h}")


# =========================
# Process the pixels CSV files one by one
# =========================
processed_count = 0

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".csv"):
        continue

    if "pixels" not in filename:
        continue

    input_path = os.path.join(input_folder, filename)

    name_without_ext, ext = os.path.splitext(filename)
    output_filename = name_without_ext + "_h_net" + ext
    output_path = os.path.join(output_folder, output_filename)

    print(f"\nProcessing file: {filename}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"  [SKIPPED] Failed to read file: {e}")
        continue

    # =========================
    # Check whether the required columns exist
    # =========================
    required_columns = ["h", "is_signal"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"  [SKIPPED] Missing required columns: {missing_columns}")
        continue

    # =========================
    # Convert to numeric so strings or empty values do not affect the calculation
    # The other columns of the original df are not modified
    # =========================
    h_numeric = pd.to_numeric(df["h"], errors="coerce")
    is_signal_numeric = pd.to_numeric(df["is_signal"], errors="coerce")

    # =========================
    # Compute h_net
    # is_signal == 1: h - negative_control_h
    # is_signal == 0: 0
    # If h_net < 0: 0
    # =========================
    df["h_net"] = 0.0

    signal_mask = is_signal_numeric == 1

    df.loc[signal_mask, "h_net"] = h_numeric.loc[signal_mask] - negative_control_h

    # Clamp negative values to 0
    df.loc[df["h_net"] < 0, "h_net"] = 0.0

    # If h itself cannot be converted to a number, set h_net to 0 as well
    df["h_net"] = df["h_net"].fillna(0.0)

    # =========================
    # Save the new file
    # The other columns of the original file are preserved
    # =========================
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        processed_count += 1
        print(f"  Saved: {output_filename}")
    except Exception as e:
        print(f"  [FAILED] Failed to save file: {e}")


print("\n==============================")
print("Processing complete")
print(f"Files processed successfully: {processed_count}")
print(f"Results saved in folder: {output_folder}")
print("==============================")