"""
Google Sheet -> University portal attendance watcher
----------------------------------------------------
Polls a Google Sheet every 5 seconds. When you edit attendance and then stop
editing (debounce), it finds only the rows that changed since the last
successful sync and hands them to the local-LLM browser agent to mark.

Runs 100% on your own computer: the LLM (Ollama), the browser and this loop.
Cron/GitHub schedules can't do 5-second checks, so this is a long-running loop;
GitHub Actions (self-hosted runner) only starts/stops it — see attendance.yml.

How the sheet is read (SHEET_MODE):
    service_account (default, recommended) - sheet stays private. Needs a free
        Google service-account key (credentials only, no paid cloud compute):
        Google Cloud Console -> enable "Google Sheets API" -> Service Account ->
        JSON key. Share the sheet with the service account's email as Viewer.
        Provide the key as service_account.json OR its full text in GOOGLE_CREDS_JSON.
    csv_link - no Google Cloud at all. Sheet must be shared as "Anyone with the
        link can view", which exposes student names to anyone who gets the link.

Settings (.env locally, or GitHub Secrets when run by the workflow):
    PORTAL_URL, PORTAL_USER, PORTAL_PASS   university portal login
    SHEET_ID        long id from the sheet URL
    WORKSHEET       tab name (service_account mode), default "Attendance"
    SHEET_GID       tab gid from the URL (csv_link mode), default 0
    STATE_DIR       where sync state + log live, default ~/attendance-agent
    RUN_FOR_SECONDS stop after this long (0 = run forever)
    OLLAMA_MODEL    default qwen2.5:14b

Expected sheet columns (row 1 = headers):
    Date | Subject | Section | Roll No | Name | Status
    Status: P / A (blank = not yet marked, ignored)
"""

import asyncio
import csv
import hashlib
import io
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from browser_use import Agent, ChatOllama  # check browser-use docs if the import path changes

load_dotenv()

POLL_SECONDS = 5           # how often to check the sheet
DEBOUNCE_SECONDS = 60      # wait until you've stopped editing for this long
BATCH_SIZE = 15            # students per agent run
RUN_FOR_SECONDS = int(os.getenv("RUN_FOR_SECONDS", "0"))
SHEET_MODE = os.getenv("SHEET_MODE", "service_account")

# Kept outside the repo folder so GitHub's checkout step can never wipe it.
STATE_DIR = Path(os.getenv("STATE_DIR", Path.home() / "attendance-agent"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "synced_state.json"   # what has already been marked on the portal
LOG_FILE = STATE_DIR / "attendance_sync.log"

STATUS_MAP = {"P": "Present", "PRESENT": "Present", "A": "Absent", "ABSENT": "Absent"}


# ---------------------------------------------------------------- Google Sheet
def make_fetcher():
    """Return a function that fetches the sheet as a list of {header: value} dicts."""
    sheet_id = os.environ["SHEET_ID"]

    if SHEET_MODE == "csv_link":
        url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}"
               f"/export?format=csv&gid={os.getenv('SHEET_GID', '0')}")

        def fetch():
            with urllib.request.urlopen(url, timeout=15) as resp:
                return list(csv.DictReader(io.StringIO(resp.read().decode("utf-8"))))
        return fetch

    import gspread
    creds_json = os.getenv("GOOGLE_CREDS_JSON")
    gc = (gspread.service_account_from_dict(json.loads(creds_json)) if creds_json
          else gspread.service_account(filename="service_account.json"))
    ws = gc.open_by_key(sheet_id).worksheet(os.getenv("WORKSHEET", "Attendance"))
    return ws.get_all_records


def read_rows(records: list) -> dict:
    """Return {(date, subject, section, roll): {name, status}} for rows with a valid status."""
    rows = {}
    for r in records:
        r = {k.strip().lower(): str(v).strip() for k, v in r.items()}
        status = STATUS_MAP.get(r.get("status", "").upper())
        if not status or not r.get("roll no"):
            continue
        key = (r.get("date", ""), r.get("subject", ""), r.get("section", ""), r["roll no"])
        rows[key] = {"name": r.get("name", ""), "status": status}
    return rows


def fingerprint(rows: dict) -> str:
    return hashlib.md5(json.dumps(sorted((list(k), v) for k, v in rows.items())).encode()).hexdigest()


# ---------------------------------------------------------------- local state
def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return {tuple(json.loads(k)): v for k, v in json.loads(STATE_FILE.read_text()).items()}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps({json.dumps(list(k)): v for k, v in state.items()}, indent=1))


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- agent
def build_task(date, subject, section, students, first, last) -> str:
    listing = "\n".join(f"- {roll} ({s['name']}): {s['status']}" for roll, s in students)
    parts = []
    if first:
        parts.append(
            "Go to x_url and log in with username x_user and password x_pass. "
            f"Open the attendance page for subject '{subject}'"
            + (f", section '{section}'" if section else "")
            + (f", date {date}" if date else ", today's date") + "."
        )
    else:
        parts.append("You are already on the attendance page. Do not reload or log in again.")
    parts.append(
        "Mark attendance for exactly these students, matching by ROLL NUMBER (not name). "
        "Do not touch any other student:\n" + listing
    )
    parts.append(
        "Then click Save/Submit once and report the confirmation message."
        if last else "Do NOT click Save/Submit yet. Report which roll numbers you marked."
    )
    parts.append("If a roll number is not on the page, skip it and list it in your report.")
    return "\n\n".join(parts)


async def mark_group(llm, date, subject, section, students) -> bool:
    sensitive = {
        "x_url": os.environ["PORTAL_URL"],
        "x_user": os.environ["PORTAL_USER"],
        "x_pass": os.environ["PORTAL_PASS"],
    }
    batches = [students[i:i + BATCH_SIZE] for i in range(0, len(students), BATCH_SIZE)]
    session = None
    for i, batch in enumerate(batches):
        agent = Agent(
            task=build_task(date, subject, section, batch, i == 0, i == len(batches) - 1),
            llm=llm,
            sensitive_data=sensitive,
            use_vision=False,
            browser_session=session,
        )
        result = await agent.run(max_steps=60)
        session = agent.browser_session
        log(f"[{date} | {subject} | {section}] batch {i + 1}/{len(batches)}: {result.final_result()}")
        if not result.is_done():
            return False
    return True


# ---------------------------------------------------------------- main loop
async def main():
    fetch = make_fetcher()
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"))
    synced = load_state()
    last_fp, last_change = None, None
    started = time.time()
    log(f"Watching sheet every {POLL_SECONDS}s (mode={SHEET_MODE}, debounce {DEBOUNCE_SECONDS}s).")

    while True:
        if RUN_FOR_SECONDS and time.time() - started > RUN_FOR_SECONDS and not last_change:
            log("Run window finished; exiting cleanly.")
            return
        try:
            current = read_rows(fetch())
            fp = fingerprint(current)

            if fp != last_fp:                       # sheet changed since last poll
                last_fp, last_change = fp, time.time()

            elif last_change and time.time() - last_change >= DEBOUNCE_SECONDS:
                last_change = None                  # editing has settled; sync once
                changed = {k: v for k, v in current.items()
                           if synced.get(k, {}).get("status") != v["status"]}
                if changed:
                    groups = defaultdict(list)
                    for (date, subj, sec, roll), v in changed.items():
                        groups[(date, subj, sec)].append((roll, v))
                    log(f"{len(changed)} changed row(s) in {len(groups)} class(es) -> marking")
                    for (date, subj, sec), students in groups.items():
                        ok = await mark_group(llm, date, subj, sec, students)
                        if ok:
                            for roll, v in students:
                                synced[(date, subj, sec, roll)] = v
                            save_state(synced)
                        else:
                            log(f"FAILED for {date} {subj} {sec}; will retry after next edit")

        except Exception as e:
            log(f"Error (backing off 30s): {e}")
            await asyncio.sleep(30)

        await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
