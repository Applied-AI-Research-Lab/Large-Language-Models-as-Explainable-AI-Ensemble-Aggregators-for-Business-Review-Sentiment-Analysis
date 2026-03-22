# python3 2_jsonl_reviews_to_csv.py
import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "review_id",
    "user_id",
    "business_id",
    "stars",
    "useful",
    "funny",
    "cool",
    "text",
    "date",
]


def convert_reviews_jsonl_to_csv(input_path: Path, output_path: Path) -> None:
    total_rows = 0
    skipped_rows = 0

    with input_path.open("r", encoding="utf-8") as json_file, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()

        for line_number, line in enumerate(json_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_rows += 1
                continue

            writer.writerow({field: record.get(field, "") for field in FIELDS})
            total_rows += 1

    print(f"CSV created: {output_path}")
    print(f"Rows written: {total_rows}")
    if skipped_rows:
        print(f"Rows skipped (invalid JSON): {skipped_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Yelp review JSONL dataset to CSV with selected columns."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("yelp_academic_dataset_review.json"),
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yelp_academic_dataset_review.csv"),
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    convert_reviews_jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()