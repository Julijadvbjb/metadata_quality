import os
import json
import requests
import pandas as pd
from typing import Any, Dict, List

BASE_URL = "https://data.gov.lv/dati/api/3/action/package_search"
ROWS = 1000

RAW_DIR = "../data_raw"
PROCESSED_DIR = "../data_processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def fetch_all_packages() -> List[Dict[str, Any]]:
    all_packages = []
    start = 0

    while True:
        url = f"{BASE_URL}?rows={ROWS}&start={start}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise RuntimeError("CKAN API atbilde nav success=True")

        result = data["result"]
        packages = result.get("results", [])

        if not packages:
            break

        all_packages.extend(packages)

        if len(packages) < ROWS:
            break

        start += ROWS

    return all_packages


def save_raw_json(packages: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(packages, f, ensure_ascii=False, indent=2)


def flatten_dict(prefix: str, obj: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, value in obj.items():
        new_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            flatten_dict(new_key, value, out)
        elif isinstance(value, list):
            out[new_key] = json.dumps(value, ensure_ascii=False)
        else:
            out[new_key] = value


def build_dataset_table(packages: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for pkg in packages:
        row = {}

        for key, value in pkg.items():
            if key in ["resources", "tags", "extras", "groups", "relationships_as_subject", "relationships_as_object"]:
                row[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                flatten_dict(key, value, row)
            else:
                row[key] = value

        row["resource_count"] = len(pkg.get("resources", []))
        row["tag_count"] = len(pkg.get("tags", []))
        row["extra_count"] = len(pkg.get("extras", []))
        row["group_count"] = len(pkg.get("groups", []))

        dataset_name = pkg.get("name")
        if dataset_name:
            row["dataset_uri"] = f"https://data.gov.lv/dati/lv/dataset/{dataset_name}"
            row["dataset_ttl_uri"] = f"https://data.gov.lv/dati/lv/dataset/{dataset_name}.ttl"
        else:
            row["dataset_uri"] = None
            row["dataset_ttl_uri"] = None

        rows.append(row)
    return pd.DataFrame(rows)

def build_extras_table(packages: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for pkg in packages:
        dataset_id = pkg.get("id")
        dataset_name = pkg.get("name")
        dataset_title = pkg.get("title")

        for extra in pkg.get("extras", []):
            rows.append({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "dataset_title": dataset_title,
                "extra_key": extra.get("key"),
                "extra_value": extra.get("value"),
            })

    return pd.DataFrame(rows)

def build_resources_table(packages: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for pkg in packages:
        dataset_id = pkg.get("id")
        dataset_name = pkg.get("name")
        dataset_title = pkg.get("title")

        for res in pkg.get("resources", []):
            row = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "dataset_title": dataset_title,
            }
            for key, value in res.items():
                if isinstance(value, dict):
                    flatten_dict(key, value, row)
                elif isinstance(value, list):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)

def build_tags_table(packages: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pkg in packages:
        dataset_id = pkg.get("id")
        dataset_name = pkg.get("name")
        dataset_title = pkg.get("title")

        for tag in pkg.get("tags", []):
            row = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "dataset_title": dataset_title,
            }
            for key, value in tag.items():
                if isinstance(value, dict):
                    flatten_dict(key, value, row)
                elif isinstance(value, list):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    packages = fetch_all_packages()
    raw_path = os.path.join(RAW_DIR, "all_packages_raw.json")
    save_raw_json(packages, raw_path)

    datasets_df = build_dataset_table(packages)
    extras_long_df = build_extras_table(packages)
    resources_df = build_resources_table(packages)
    tags_df = build_tags_table(packages)

    datasets_df.to_csv(os.path.join(PROCESSED_DIR, "datasets_all_fields.csv"), index=False, encoding="utf-8-sig")
    extras_long_df.to_csv(os.path.join(PROCESSED_DIR, "extras.csv"), index=False, encoding="utf-8-sig")
    resources_df.to_csv(os.path.join(PROCESSED_DIR, "resources.csv"), index=False, encoding="utf-8-sig")
    tags_df.to_csv(os.path.join(PROCESSED_DIR, "tags.csv"), index=False, encoding="utf-8-sig")

    excel_path = os.path.join(PROCESSED_DIR, "metadata_full_export.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        datasets_df.to_excel(writer, sheet_name="datasets_all_fields", index=False)
        extras_long_df.to_excel(writer, sheet_name="extras_long", index=False)
        resources_df.to_excel(writer, sheet_name="resources_all_fields", index=False)
        tags_df.to_excel(writer, sheet_name="tags_all_fields", index=False)

if __name__ == "__main__":
    main()