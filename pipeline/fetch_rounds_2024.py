"""
Fetch round-level SG data for all 2024 PGA events.
Respects DataGolf rate limits: 24 requests per 5 min window.
"""

import json
import os
import time
import urllib.request
import urllib.error

API_KEY = "85acc5b95e8b7ead122cab0c8020"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_LIST = os.path.join(PROJECT_ROOT, "data", "raw", "dg_event_list_pga.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "rounds")
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 24
COOLDOWN_SECS = 310  # 5 min + buffer


def fetch(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", COOLDOWN_SECS))
                print(f"    429 rate limited - waiting {retry_after}s...")
                time.sleep(retry_after)
            else:
                raise
    return None


def main():
    # Load event list, filter to 2024 + sg_categories=yes
    with open(EVENT_LIST) as f:
        events = json.load(f)

    events_2024 = [
        e for e in events
        if e["calendar_year"] == 2024 and e["sg_categories"] == "yes"
    ]
    print(f"Found {len(events_2024)} events in 2024 with SG data")

    # Check which are already cached
    to_fetch = []
    already_cached = 0
    for e in events_2024:
        out_path = os.path.join(OUT_DIR, f"2024_{e['event_id']}.json")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 200:
            already_cached += 1
        else:
            to_fetch.append(e)

    print(f"Already cached: {already_cached}")
    print(f"To fetch: {len(to_fetch)}")

    if not to_fetch:
        print("Nothing to fetch!")
    else:
        for i, e in enumerate(to_fetch):
            # Rate limit: pause between batches
            if i > 0 and i % BATCH_SIZE == 0:
                print(f"\n  Batch limit reached. Cooling down {COOLDOWN_SECS}s...")
                time.sleep(COOLDOWN_SECS)

            eid = e["event_id"]
            url = (
                f"https://feeds.datagolf.com/historical-raw-data/rounds"
                f"?tour=pga&event_id={eid}&year=2024&file_format=json&key={API_KEY}"
            )
            out_path = os.path.join(OUT_DIR, f"2024_{eid}.json")

            print(f"  [{i+1}/{len(to_fetch)}] {e['event_name']} (id={eid})...", end=" ", flush=True)
            data = fetch(url)

            if data:
                with open(out_path, "w") as f:
                    json.dump(data, f)
                n_players = len(data.get("scores", []))
                print(f"{n_players} players")
            else:
                print("FAILED")

    # Summary: count total player-rounds
    total_events = 0
    total_player_rounds = 0
    for e in events_2024:
        out_path = os.path.join(OUT_DIR, f"2024_{e['event_id']}.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)
            scores = data.get("scores", [])
            total_events += 1
            for player in scores:
                for rkey in ("round_1", "round_2", "round_3", "round_4"):
                    if rkey in player and player[rkey] is not None:
                        total_player_rounds += 1

    print(f"\n{'='*50}")
    print(f"Total events cached:       {total_events}")
    print(f"Total player-rounds:       {total_player_rounds}")
    print(f"Avg rounds per event:      {total_player_rounds / total_events:.0f}")


if __name__ == "__main__":
    main()
