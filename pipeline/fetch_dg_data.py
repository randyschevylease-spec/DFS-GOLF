import json
import os
import csv
import time
import urllib.request
import urllib.error

API_KEY = "85acc5b95e8b7ead122cab0c8020"
BASE_URL = "https://feeds.datagolf.com/historical-dfs-data"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(OUT_DIR, "raw")
os.makedirs(CACHE_DIR, exist_ok=True)

# Rate limit: ~26 requests per 5 min window, 300s cooldown
BATCH_SIZE = 24         # stay under the 26 limit
COOLDOWN_SECS = 310     # 5 min + buffer

def fetch(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())

def fetch_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fetch(url)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", COOLDOWN_SECS))
                print(f"    429 - waiting {retry_after}s...")
                time.sleep(retry_after + 5)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")

# --- Load event list ---
EVENT_CACHE = os.path.join(CACHE_DIR, "event_list.json")
if os.path.exists(EVENT_CACHE):
    print("Loading event list from cache...")
    with open(EVENT_CACHE, "r") as f:
        events = json.load(f)
else:
    print("Fetching event list...")
    events = fetch_with_retry(f"{BASE_URL}/event-list?file_format=json&key={API_KEY}")
    with open(EVENT_CACHE, "w") as f:
        json.dump(events, f)
print(f"Found {len(events)} events")

# --- Build fetch list ---
fetch_list = []
for e in events:
    if e["dk_salaries"] == "yes":
        fetch_list.append((e, "draftkings"))
    if e["fd_salaries"] == "yes":
        fetch_list.append((e, "fanduel"))

# Split into cached vs uncached
uncached = []
for item in fetch_list:
    e, site = item
    cache_file = os.path.join(CACHE_DIR, f"{e['tour']}_{e['event_id']}_{e['calendar_year']}_{site}.json")
    if not os.path.exists(cache_file):
        uncached.append(item)

print(f"Total: {len(fetch_list)} | Already cached: {len(fetch_list) - len(uncached)} | To fetch: {len(uncached)}")
if uncached:
    batches = (len(uncached) + BATCH_SIZE - 1) // BATCH_SIZE
    est_min = batches * COOLDOWN_SECS / 60
    print(f"Estimated time: {batches} batches x 5min = ~{est_min:.0f} min")

# --- Fetch uncached data in batches ---
api_calls = 0
for i, (event, site) in enumerate(uncached):
    eid = event["event_id"]
    year = event["calendar_year"]
    tour = event["tour"]
    cache_file = os.path.join(CACHE_DIR, f"{tour}_{eid}_{year}_{site}.json")

    url = (
        f"{BASE_URL}/points?tour={tour}&event_id={eid}&year={year}"
        f"&site={site}&file_format=json&key={API_KEY}"
    )

    try:
        data = fetch_with_retry(url)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        api_calls += 1
        n = len(data.get("dfs_points", []))
        print(f"  [{i+1}/{len(uncached)}] {site:<11} | {event['event_name'][:45]:<45} | {event['date']} | {n} players")
    except Exception as ex:
        print(f"  [{i+1}/{len(uncached)}] ERROR: {event['event_name']} ({site}) - {ex}")
        continue

    # Batch pacing: after every BATCH_SIZE calls, cool down
    if api_calls > 0 and api_calls % BATCH_SIZE == 0 and i < len(uncached) - 1:
        print(f"\n  --- Batch done ({api_calls} calls). Cooling down {COOLDOWN_SECS}s... ---\n")
        time.sleep(COOLDOWN_SECS)

# --- Build CSV from all cached data ---
print("\nBuilding CSV from all cached data...")
all_rows = []
for event, site in fetch_list:
    eid = event["event_id"]
    year = event["calendar_year"]
    tour = event["tour"]
    cache_file = os.path.join(CACHE_DIR, f"{tour}_{eid}_{year}_{site}.json")

    if not os.path.exists(cache_file):
        continue

    with open(cache_file, "r") as f:
        data = json.load(f)

    for p in data.get("dfs_points", []):
        row = {
            "calendar_year": year,
            "date": event["date"],
            "event_name": event["event_name"],
            "event_id": eid,
            "tour": tour,
            "site": site,
            "dk_ownerships_available": event["dk_ownerships"],
            "fd_ownerships_available": event["fd_ownerships"],
        }
        row.update(p)
        all_rows.append(row)

# Ordered columns
meta_cols = [
    "calendar_year", "date", "event_name", "event_id", "tour", "site",
    "dk_ownerships_available", "fd_ownerships_available",
    "dg_id", "player_name", "fin_text", "salary", "ownership", "total_pts",
]
all_keys = list(dict.fromkeys(meta_cols))
for row in all_rows:
    for k in row:
        if k not in all_keys:
            all_keys.append(k)

os.makedirs(os.path.join(OUT_DIR, "processed"), exist_ok=True)
out_path = os.path.join(OUT_DIR, "processed", "historical_dfs_all.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nDone! Wrote {len(all_rows)} rows to {out_path}")
remaining = len(uncached) - api_calls
if remaining > 0:
    print(f"WARNING: {remaining} events still missing. Re-run to continue.")
