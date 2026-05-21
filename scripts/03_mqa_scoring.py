
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
import zipfile
from urllib.parse import urlparse

DATASETS_PATH = "../data_processed/datasets_all_fields.csv"
RESOURCES_PATH = "../data_processed/resources_all_fields.csv"
EXTRAS_PATH = "../data_processed/extras_long.csv"
OUTPUT_PATH = "../outputs/mqa_scores.csv"
URL_CACHE_PATH = "../outputs/url_check_cache.csv"
PROGRESS_PATH = "../outputs/url_check_progress.txt"

CHECK_URLS = True
URL_TIMEOUT = 8
MAX_WORKERS = 20

NON_PROPRIETARY = {
    "CSV", "TSV", "JSON", "JSONLD", "JSON-LD", "JSON:API",
    "XML", "RDF", "RDFXML", "RDF/XML", "TTL", "TURTLE", "N3",
    "NTRIPLES", "NQUADS", "GEOJSON", "GPKG", "GEOPACKAGE", "KML",
    "KMZ", "GML", "ATOM", "NETCDF", "NC", "GRIB", "GEOTIFF",
    "HTML", "HTM", "XHTML", "XHTML+XML", "TXT", "MD", "MARKDOWN",
    "YAML", "YML", "ODS", "ODT", "ODP", "EPUB", "PDF", "PARQUET",
    "AVRO", "ORC", "PNG", "JPG", "JPEG", "SVG", "TIFF", "TIF", "GIF",
    "BMP", "MP3", "MP4", "OGG", "WAV", "WEBM", "CITYGML", "OBJ", "3DS", "SHP",
}

PROPRIETARY = {
    "XLS", "XLSX", "DOC", "DOCX", "PPT", "PPTX", "MDB", "ACCDB", "GDB",
    "DWG", "DXF", "AI", "PSD", "MXD", "SKP", "RAR",
}

MACHINE_READABLE = {
    "CSV", "TSV", "JSON", "JSONLD", "JSON-LD", "JSON:API", "XML", "RDF",
    "RDFXML", "RDF/XML", "TTL", "TURTLE", "N3", "NTRIPLES", "NQUADS",
    "GEOJSON", "GPKG", "GEOPACKAGE", "KML", "KMZ", "GML", "ATOM",
    "NETCDF", "NC", "GRIB", "GEOTIFF", "XLS", "XLSX", "ODS", "MDB",
    "ACCDB", "SHP", "GDB", "PARQUET", "AVRO", "ORC", "YAML", "YML", "CITYGML",
}
ARCHIVE_FORMATS = {"ZIP", "GZ", "TAR", "7Z"}

KNOWN_FORMATS = NON_PROPRIETARY | PROPRIETARY | MACHINE_READABLE | ARCHIVE_FORMATS

LICENSE_VOCAB = {
    "CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0",
    "CC-BY-ND-4.0", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "PDDL-1.0",
    "ODbL-1.0", "ODC-BY-1.0", "OGL-3.0", "EUPL-1.2", "MIT",
    "Apache-2.0", "GPL-3.0", "BSD-3-Clause",
}

ACCESS_RIGHTS_VOCAB = {
    "PUBLIC", "NON_PUBLIC", "RESTRICTED", "SENSITIVE", "CONFIDENTIAL",
    "HTTP://PUBLICATIONS.EUROPA.EU/RESOURCE/AUTHORITY/ACCESS-RIGHT/PUBLIC",
    "HTTP://PUBLICATIONS.EUROPA.EU/RESOURCE/AUTHORITY/ACCESS-RIGHT/NON_PUBLIC",
    "HTTP://PUBLICATIONS.EUROPA.EU/RESOURCE/AUTHORITY/ACCESS-RIGHT/RESTRICTED",
}


def has_value(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip()
    if not s:
        return False
    return s.lower() not in {"[]", "{}", "null", "none", "nan"}

def extension_from_url(url: str) -> str | None:
    try:
        path = urlparse(url).path
        if "." not in path:
            return None
        return path.split(".")[-1].upper()
    except Exception:
        return None
    
def norm_format(f) -> Optional[str]:
    if not has_value(f):
        return None
    return str(f).upper().strip().lstrip(".")


def norm_key(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).lower().strip().replace("-", "_").replace(" ", "_")


def norm_text(s) -> str:
    if not has_value(s):
        return ""
    return str(s).strip().upper()

def _make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "Mozilla/5.0 (MQA-research-bot)"})
    return sess

_session = _make_session()

def inspect_zip_content(url: str) -> dict:
    result = {
        "zip_contains_machine_readable": 0,
        "zip_file_extensions": "",
    }

    try:
        r = _session.get(url, timeout=URL_TIMEOUT, stream=True)
        if not (200 <= r.status_code < 400):
            r.close()
            return result

        content = r.content
        r.close()

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            extensions = set()

            for name in zf.namelist():
                if "." in name:
                    ext = name.split(".")[-1].upper()
                    extensions.add(ext)

            result["zip_file_extensions"] = ", ".join(sorted(extensions))

            if any(ext in MACHINE_READABLE for ext in extensions):
                result["zip_contains_machine_readable"] = 1

    except Exception:
        pass

    return result

def _check_one_url(url: str) -> dict:
    u = url.strip()

    result = {
        "url": u,
        "accessible": 0,
        "downloadable": 0,
        "status_code": None,
        "content_type": None,
        "content_length": None,
        "is_html": 0,
        "zip_contains_machine_readable": 0,
        "zip_file_extensions": "",
    }

    try:
        r = _session.head(u, timeout=URL_TIMEOUT, allow_redirects=True)

        if r.status_code in {400, 403, 405, 501}:
            r = _session.get(
                u,
                timeout=URL_TIMEOUT,
                allow_redirects=True,
                stream=True,
            )

        result["status_code"] = r.status_code
        result["content_type"] = r.headers.get("Content-Type", "")
        result["content_length"] = r.headers.get("Content-Length")

        if 200 <= r.status_code < 400:
            result["accessible"] = 1

            content_type = result["content_type"].lower()
            content_disposition = r.headers.get("Content-Disposition", "").lower()

            if "text/html" in content_type:
                result["is_html"] = 1

            if "attachment" in content_disposition or "text/html" not in content_type:
                result["downloadable"] = 1

            ext = extension_from_url(u)
            if ext == "ZIP" and result["downloadable"] == 1:
                zip_info = inspect_zip_content(u)
                result.update(zip_info)

        r.close()

    except requests.RequestException:
        pass

    return result

def build_url_cache(all_urls: list[str]) -> dict[str, dict]:
    cache: dict[str, dict] = {}

    if os.path.exists(URL_CACHE_PATH):
        old = pd.read_csv(URL_CACHE_PATH)
        for _, row in old.iterrows():
            cache[str(row["url"])] = row.to_dict()

    unique = sorted({u.strip() for u in all_urls if has_value(u)})
    to_check = [u for u in unique if u not in cache]

    if not to_check:
        return cache

    done = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_url = {
            pool.submit(_check_one_url, u): u
            for u in to_check
        }

        for fut in as_completed(future_to_url):
            url = future_to_url[fut]

            try:
                cache[url] = fut.result()
            except Exception:
                cache[url] = {
                    "url": url,
                    "accessible": 0,
                    "downloadable": 0,
                    "zip_contains_machine_readable": 0,
                    "zip_file_extensions": "",
                }

            done += 1

            if done % 50 == 0 or done == len(to_check):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(to_check) - done) / rate if rate > 0 else 0

                msg = (
                    f"URL pārbaude: {done}/{len(to_check)} | "
                    f"{rate:.1f}/s | ETA {eta/60:.1f} min"
                )
                print(msg)

                pd.DataFrame(cache.values()).to_csv(
                    URL_CACHE_PATH,
                    index=False,
                    encoding="utf-8-sig",
                )

    pd.DataFrame(cache.values()).to_csv(
        URL_CACHE_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    return cache

def extra_has_keyword(ext: dict[str, str], keywords: Iterable[str]) -> int:
    for key in ext:
        if any(keyword in key for keyword in keywords):
            return 1
    return 0


def extra_value_for_keywords(ext: dict[str, str], keywords: Iterable[str]) -> Optional[str]:
    for key, value in ext.items():
        if any(keyword in key for keyword in keywords):
            return value
    return None

def resource_flags(res_subset: pd.DataFrame, url_cache: dict[str, dict]) -> dict[str, int]:
    if res_subset.empty:
        return {
            "url_accessible": 0,
            "url_downloadable": 0,
            "has_format": 0,
            "has_mimetype": 0,
            "known_vocab": 0,
            "non_proprietary": 0,
            "machine_readable": 0,
            "byte_size": 0,
            "zip_contains_machine_readable": 0,
        }

    urls = [
        str(u).strip()
        for u in res_subset.get("url", pd.Series(dtype=object)).tolist()
        if has_value(u)
    ]

    formats = [
        norm_format(f)
        for f in res_subset.get("format", pd.Series(dtype=object)).tolist()
    ]
    formats = [f for f in formats if f is not None]

    url_records = [url_cache.get(u, {}) for u in urls]

    url_accessible = int(
        any(rec.get("accessible", 0) == 1 for rec in url_records)
    ) if CHECK_URLS else 0

    url_downloadable = int(
        any(rec.get("downloadable", 0) == 1 for rec in url_records)
    ) if CHECK_URLS else 0

    zip_contains_machine_readable = int(
        any(rec.get("zip_contains_machine_readable", 0) == 1 for rec in url_records)
    ) if CHECK_URLS else 0

    machine_readable = int(
        any(f in MACHINE_READABLE for f in formats)
        or zip_contains_machine_readable == 1
    )

    return {
        "has_url": int(len(urls) > 0),
        "url_accessible": url_accessible,
        "url_downloadable": url_downloadable,
        "has_format": int(len(formats) > 0),
        "has_mimetype": int(
            res_subset.get("mimetype", pd.Series(dtype=object))
            .apply(has_value)
            .any()
        ),
        "known_vocab": int(any(f in KNOWN_FORMATS for f in formats)),
        "non_proprietary": int(any(f in NON_PROPRIETARY for f in formats)),
        "machine_readable": machine_readable,
        "byte_size": int(
            res_subset.get("size", pd.Series(dtype=object))
            .apply(has_value)
            .any()
        ),
        "zip_contains_machine_readable": zip_contains_machine_readable,
    }

def quality_label(total_points: int) -> str:
    if total_points >= 351:
        return "Excellent"
    if total_points >= 221:
        return "Good"
    if total_points >= 121:
        return "Sufficient"
    return "Bad"

def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    datasets = pd.read_csv(DATASETS_PATH)
    resources = pd.read_csv(RESOURCES_PATH)
    extras = pd.read_csv(EXTRAS_PATH)

    url_cache: dict[str, int] = {}
    if CHECK_URLS:
        all_urls = resources["url"].dropna().astype(str).tolist()
        url_cache = build_url_cache(all_urls)
        pct_ok = (
            sum(rec.get("accessible", 0) for rec in url_cache.values())/ len(url_cache) * 100 if url_cache else 0
        )
    extras_dict: dict[str, dict[str, str]] = {}
    for _, row in extras.iterrows():
        dataset_id = row["dataset_id"]
        key = norm_key(row["extra_key"])
        value = row["extra_value"]
        if has_value(value):
            extras_dict.setdefault(dataset_id, {})[key] = value

    res_by_dataset = {ds_id: grp for ds_id, grp in resources.groupby("dataset_id")}

    results = []
    n = len(datasets)
    for i, (_, row) in enumerate(datasets.iterrows(), start=1):

        dataset_id = row["id"]
        ext = extras_dict.get(dataset_id, {})
        res = res_by_dataset.get(dataset_id, resources.iloc[0:0])
        flags = resource_flags(res, url_cache)
        keyword = int(has_value(row.get("tag_count")) and row.get("tag_count") > 0)
        theme = extra_has_keyword(ext, ["theme"])
        spatial = extra_has_keyword(ext, ["spatial", "bbox"])
        temporal = extra_has_keyword(ext, ["temporal"])
        findability = keyword * 30 + theme * 30 + spatial * 20 + temporal * 20

        access_url_accessible = flags["url_accessible"]
        download_url_present = flags["has_url"]
        download_url_accessible = flags["url_downloadable"]
        accessibility = (
            access_url_accessible * 50
            + download_url_present * 20
            + download_url_accessible * 30
        )

        if "dcat_ap_compliant" in row.index:
            dcat_ap = int(str(row.get("dcat_ap_compliant")).strip().lower() in {"1", "true", "yes", "jā"})
        else:
            dcat_ap = 0
        interoperability = (
            flags["has_format"] * 20
            + flags["has_mimetype"] * 10
            + flags["known_vocab"] * 10
            + flags["non_proprietary"] * 20
            + flags["machine_readable"] * 20
            + dcat_ap * 30
        )

        license_id = row.get("license_id")
        license_present = int(has_value(license_id))
        license_vocab = int(license_present and norm_text(license_id) in LICENSE_VOCAB)
        access_rights_value = extra_value_for_keywords(ext, ["access_constraints", "access_rights"])
        access_rights = int(has_value(access_rights_value))
        access_rights_vocab = int(norm_text(access_rights_value) in ACCESS_RIGHTS_VOCAB)
        contact = int(
            has_value(row.get("maintainer_email"))
            or has_value(row.get("author_email"))
            or extra_has_keyword(ext, ["contact_email", "contact", "responsible_party"])
        )
        publisher = int(has_value(row.get("organization_title")))
        reusability = (
            license_present * 20
            + license_vocab * 10
            + access_rights * 10
            + access_rights_vocab * 5
            + contact * 20
            + publisher * 10
        )

        rights = extra_has_keyword(ext, ["rights"])
        issued = int(has_value(row.get("metadata_created")))
        modified = int(has_value(row.get("metadata_modified")))
        contextuality = rights * 5 + flags["byte_size"] * 5 + issued * 5 + modified * 5

        total = int(findability + accessibility + interoperability + reusability + contextuality)
        percent = round(total / 405 * 100, 2)

        results.append({
            "dataset_id": dataset_id,
            "title": row.get("title"),
            "organization_title": row.get("organization_title"),
            "Findability": findability,
            "Accessibility": accessibility,
            "Interoperability": interoperability,
            "Reusability": reusability,
            "Contextuality": contextuality,
            "MQA_total": total,
            "MQA_percent": percent,
            "MQA_rating": quality_label(total),
            "I_keyword": keyword,
            "I_theme": theme,
            "I_spatial": spatial,
            "I_temporal": temporal,
            "I_accessURL_accessible": access_url_accessible,
            "I_downloadURL": download_url_present,
            "I_downloadURL_accessible": download_url_accessible,
            "I_format": flags["has_format"],
            "I_mimetype": flags["has_mimetype"],
            "I_format_media_vocab": flags["known_vocab"],
            "I_nonproprietary": flags["non_proprietary"],
            "I_machine_readable": flags["machine_readable"],
            "I_dcat_ap_compliance": dcat_ap,
            "I_license": license_present,
            "I_license_vocab": license_vocab,
            "I_access_rights": access_rights,
            "I_access_rights_vocab": access_rights_vocab,
            "I_contact_point": contact,
            "I_publisher": publisher,
            "I_rights": rights,
            "I_byte_size": flags["byte_size"],
            "I_issued": issued,
            "I_modified": modified,
            "I_downloadable": flags["url_downloadable"],
            "I_zip_contains_machine_readable": flags["zip_contains_machine_readable"],
        })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
