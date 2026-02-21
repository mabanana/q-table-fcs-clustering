import os
import shutil
import pandas as pd
import fcsparser


def load_labels_dict(data_dir="./data/AML/"):
    """Load sample labels from AML.csv into a dictionary."""
    csv_path = os.path.join(data_dir, "AML.csv")
    df = pd.read_csv(csv_path)
    labels_dict = {
        str(int(fcs_file_name)): str(label).strip().lower()
        for fcs_file_name, label in zip(df['FCSFileName'], df['Label'])
    }
    return labels_dict


def _clear_directory(directory_path: str):
    """Remove all files/subdirectories in a target directory."""
    os.makedirs(directory_path, exist_ok=True)
    for entry_name in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry_name)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)


def _extract_numeric_key(filename: str) -> str:
    """Extract integer key from filename like 0001.FCS -> '1'."""
    stem = os.path.splitext(filename)[0]
    return str(int(stem))


def split_aml_data(
    aml_fcs_dir: str = "./data/AML/FCS",
    positive_dir: str = "./data/positive",
    mixed_training_dir: str = "./data/mixed_training",
    mixed_dir: str = "./data/mixed",
):
    """
    Split AML files in half and populate project data folders.

    Rules:
    - First half: all files copied to mixed_training_dir with label suffix
      (e.g. 0001_negative.FCS, 0033_positive.FCS)
    - First half: files labeled 'aml' are also copied to positive_dir
    - Second half: all files copied to mixed_dir (original filename)
    """
    labels_dict = load_labels_dict("./data/AML/")

    all_fcs_files = sorted(
        [file_name for file_name in os.listdir(aml_fcs_dir) if file_name.lower().endswith(".fcs")]
    )

    if not all_fcs_files:
        raise ValueError(f"No FCS files found in {aml_fcs_dir}")

    midpoint = len(all_fcs_files) // 2
    first_half = all_fcs_files[:midpoint]
    second_half = all_fcs_files[midpoint:]

    _clear_directory(positive_dir)
    _clear_directory(mixed_training_dir)
    _clear_directory(mixed_dir)

    train_total = 0
    train_aml = 0
    test_total = 0
    missing_labels = 0

    for file_name in first_half:
        source_path = os.path.join(aml_fcs_dir, file_name)
        numeric_key = _extract_numeric_key(file_name)
        label = labels_dict.get(numeric_key)

        if label is None:
            missing_labels += 1
            continue

        label_suffix = "positive" if label == "aml" else "negative"
        training_filename = f"{os.path.splitext(file_name)[0]}_{label_suffix}.FCS"
        training_path = os.path.join(mixed_training_dir, training_filename)
        shutil.copy2(source_path, training_path)
        train_total += 1

        if label == "aml":
            positive_path = os.path.join(positive_dir, file_name)
            shutil.copy2(source_path, positive_path)
            train_aml += 1

    for file_name in second_half:
        source_path = os.path.join(aml_fcs_dir, file_name)
        target_path = os.path.join(mixed_dir, file_name)
        shutil.copy2(source_path, target_path)
        test_total += 1

    print("Data split completed.")
    print(f"  Total source files: {len(all_fcs_files)}")
    print(f"  First half (mixed_training): {train_total}")
    print(f"  AML in first half copied to positive: {train_aml}")
    print(f"  Second half (mixed): {test_total}")
    print(f"  Missing labels skipped in training half: {missing_labels}")


def print_headers():
    """Print the headers of the AML.csv file."""
    csv_path = os.path.join("./data/AML/FCS/", "0001.FCS")
    _, data = fcsparser.parse(csv_path, reformat_meta=True)
    print("Headers in the FCS file:")
    for column in data.columns:
        print(column)


if __name__ == "__main__":
    split_aml_data()
    print_headers()

