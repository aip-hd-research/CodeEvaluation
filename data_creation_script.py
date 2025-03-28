import os
import csv


def create_csv_from_d_files(directory: str, output_csv: str):
    """
    Reads all .d files in the given directory and
    creates a CSV file with the file content.

    Args:
        directory (str): Path to the directory containing .d files.
        output_csv (str): Path to save the output CSV file.
    """

    data = []

    # Read all .d files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".d"):
            file_path = os.path.join(directory, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            file_id = os.path.splitext(filename)[0]  # Remove .d extension
            data.append({"id": file_id, "d_with_params": content})

    if not data:
        raise ValueError("No .d files found in the specified directory.")

    # Write to CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "d_with_params"])
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV saved at {output_csv}")


# Example usage
create_csv_from_d_files(
    "/nfs/home/jschmid_msc2024/dev/my-rome/d_running/data/d_with_params",
    "d_with_params.csv",
)
