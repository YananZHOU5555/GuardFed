"""Bounded public-source recovery; saves raw responses and distinguishes metadata.

Does not use credentials, contact authors, or circumvent access controls.
"""
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

HERE = Path(__file__).resolve().parent
OUT = HERE / "sources" / "recovery_20261009"
OUT.mkdir(exist_ok=True)
DOIS = {
    "fedwa": "10.5555/3535850.3536051",
    "smartfl": "10.1016/j.inffus.2025.103555",
    "feddna": "10.1016/j.jisa.2025.104358",
}
URLS = []
for name, doi in DOIS.items():
    URLS.extend([
        (name + "_s2", "https://api.semanticscholar.org/graph/v1/paper/DOI:" + doi + "?fields=title,authors,externalIds,openAccessPdf,url", "metadata"),
        (name + "_crossref", "https://api.crossref.org/works/" + doi, "metadata"),
        # Missing email is recorded as such, not replaced with a made-up contact.
        (name + "_unpaywall", "https://api.unpaywall.org/v2/" + doi, "metadata_request"),
    ])
URLS.extend([
    ("smartfl_author_home", "https://cobbd.github.io/", "author_homepage"),
    ("smartfl_author_repos", "https://api.github.com/users/cobbd/repos?per_page=100", "repository_metadata"),
    ("smartfl_uwa", "https://research-repository.uwa.edu.au/en/publications/smartfl-simple-majority-rule-based-byzantine-robust-federated-lea/", "institutional_metadata"),
    ("fedwa_openalex", "https://api.openalex.org/works/https://doi.org/10.5555/3535850.3536051", "metadata"),
])


def fetch(item):
    name, url, kind = item
    rec = {"name": name, "url": url, "kind": kind, "checked_utc": datetime.now(timezone.utc).isoformat()}
    try:
        try:
            response = urlopen(Request(url, headers={"User-Agent": "GuardFed-source-verification/1.0"}), timeout=22)
        except HTTPError as e:
            response = e
        with response:
            content = response.read()
            status = response.status
            final_url = response.url
            content_type = response.headers.get("Content-Type", "")
        suffix = ".json" if "json" in content_type else ".html"
        p = OUT / (name + suffix)
        p.write_bytes(content)
        rec.update(status_code=status, final_url=final_url, content_type=content_type, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), saved_as=p.name)
        if status == 200 and suffix == ".json":
            d = json.loads(content)
            if name.endswith("_s2"):
                rec.update(title=d.get("title"), externalIds=d.get("externalIds"), openAccessPdf=d.get("openAccessPdf"))
            elif name.endswith("_openalex"):
                rec.update(title=d.get("title"), open_access=d.get("open_access"), locations=d.get("locations"))
            elif name.endswith("_crossref"):
                rec.update(title=d.get("message", {}).get("title"), links=d.get("message", {}).get("link"))
            elif name.endswith("_repos"):
                rec["repositories"] = [{"name": a["full_name"], "url": a["html_url"], "description": a.get("description")} for a in d]
        if name.endswith("_unpaywall") and status == 422:
            rec["limitation"] = "Unpaywall API requires a genuine contact email; none supplied and no author email impersonated."
        rec["fulltext_obtained"] = False
    except (URLError, TimeoutError, OSError) as e:
        rec.update(error=type(e).__name__ + ": " + str(e), fulltext_obtained=False)
    return rec


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(fetch, URLS))
    (OUT / "api_receipts.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))
