"""Save ordinary public GET responses once; no credentials or access workaround."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={"User-Agent": "GuardFed-public-source-check/1.0", "Accept": "application/json,text/html,application/pdf;q=0.9,*/*;q=0.8"})
    result = {"name": name, "url": url, "checked_utc": datetime.now(timezone.utc).isoformat(), "fulltext_obtained": False}
    try:
        response = urllib.request.urlopen(request, timeout=25)
    except urllib.error.HTTPError as exc:
        response = exc
    except Exception as exc:
        result["error"] = repr(exc)
        return result
    with response:
        data = response.read()
        ctype = response.headers.get("Content-Type", "")
        suffix = ".json" if "json" in ctype else ".pdf" if data.startswith(b"%PDF-") else ".html"
        RAW.mkdir(parents=True, exist_ok=True)
        path = RAW / (name + suffix)
        if path.exists():
            raise FileExistsError("Do not repeat an already saved response: " + name)
        path.write_bytes(data)
        result.update(status_code=response.status, final_url=response.url, content_type=ctype,
                      bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), saved_as="raw/" + path.name)
    return result


def batch(items, receipt):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, items))
    (ROOT / receipt).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps([{k: r.get(k) for k in ["name", "status_code", "bytes", "final_url", "error"]} for r in results], ensure_ascii=False))
    return results
