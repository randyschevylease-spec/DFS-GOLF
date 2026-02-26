"""Google Sheets Export — Push lineups and projections to Google Sheets.

Usage:
    from google_sheets import export_to_sheets

    url = export_to_sheets(
        event_name="The Genesis Invitational",
        lineups=lineups,              # list of lineup dicts from optimizer
        projected_players=players,    # full player projections list
        contest_profile=profile,      # optional contest profile dict
        contest_metrics=metrics,      # optional contest metrics dict
        contest_params=params,        # optional contest params dict
    )
"""
import re
import os
import gspread
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

CREDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
SHEET_ID = "12YBOTHuRWX6EK6mGz6Uu-VfstwtRWuHlGwl3Zg5UHbc"

HEADER_FMT = {
    "backgroundColor": {"red": 0.15, "green": 0.15, "blue": 0.15},
    "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": True},
    "horizontalAlignment": "CENTER",
}


def _get_client():
    """Authenticate and return a gspread client."""
    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    return gspread.authorize(creds)


def _get_or_create_worksheet(spreadsheet, title, rows, cols):
    """Get existing worksheet by title, or create it. Clears if it exists."""
    try:
        ws = spreadsheet.worksheet(title)
        ws.clear()
        try:
            ws.clear_basic_filter()
        except Exception:
            pass
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
    return ws


def _contest_prefix(contest_profile):
    """Derive a short tab prefix from contest name.

    Examples:
        'PGA TOUR $1M Sand Trap [$200K to 1st]' → '1M'
        'PGA TOUR $300K Drive the Green'         → '300K'
        'PGA TOUR $75K Long Drive (SE)'          → '75K SE'
    """
    if not contest_profile:
        return ""

    name = contest_profile.get("name", "")

    # Extract dollar amount (e.g., $1M, $300K, $75K)
    m = re.search(r'\$(\d+(?:\.\d+)?[MKmk]?)', name)
    prefix = m.group(1).upper() if m else ""

    # Add SE tag for single entry
    max_per_user = contest_profile.get("max_entries_per_user", 0)
    if max_per_user == 1:
        prefix += " SE"

    return prefix.strip()


def export_to_sheets(event_name, lineups, projected_players,
                     contest_profile=None, contest_metrics=None,
                     contest_params=None, sim_results=None,
                     sim_summary=None, portfolio_analytics=None):
    """Export optimizer output to the shared Google Sheet.

    Each contest gets its own set of tabs, prefixed by contest name
    (e.g., '1M | Lineups', '300K | Lineups', '75K SE | Lineups').

    Returns the spreadsheet URL.
    """
    client = _get_client()
    spreadsheet = client.open_by_key(SHEET_ID)

    # Update spreadsheet title to the event name
    spreadsheet.update_title(f"DFS Golf — {event_name}")

    # Derive tab prefix from contest
    prefix = _contest_prefix(contest_profile)
    if not prefix:
        prefix = f"{len(lineups)}L"  # fallback: "150L", "1L"

    def tab(name):
        return f"{prefix} | {name}"

    # ── Tab 1: Lineups (DK upload format) ──
    lineup_ws = _get_or_create_worksheet(spreadsheet, tab("Lineups"), len(lineups) + 1, 6)

    lineup_rows = [["G", "G", "G", "G", "G", "G"]]
    for lineup in lineups:
        row = [p.get("name_id", p["name"]) for p in lineup]
        lineup_rows.append(row)

    lineup_ws.update(lineup_rows, value_input_option="RAW")
    lineup_ws.format("A1:F1", HEADER_FMT)

    # ── Tab 2: Projections ──
    proj_ws = _get_or_create_worksheet(spreadsheet, tab("Projections"), len(projected_players) + 1, 12)

    proj_header = [
        "Player", "Salary", "Proj Pts", "Value", "MC%",
        "Own%", "Std Dev", "Wave", "Wave Adj", "Move Adj",
        "Scoring Pts", "Finish Pts",
    ]
    proj_rows = [proj_header]
    for p in projected_players:
        mc_pct = round(p.get("p_make_cut", 0) * 100, 1) if p.get("p_make_cut") else 0
        proj_rows.append([
            p["name"],
            p["salary"],
            p["projected_points"],
            p.get("value", 0),
            mc_pct,
            p.get("proj_ownership", 0),
            p.get("std_dev", 0),
            p.get("wave", "?"),
            p.get("wave_adj", 0),
            p.get("movement_adj", 0),
            p.get("proj_points_scoring", 0),
            p.get("proj_points_finish", 0),
        ])

    proj_ws.update(proj_rows, value_input_option="RAW")
    proj_ws.format("A1:L1", HEADER_FMT)

    # ── Tab 3: Lineup Detail ──
    detail_rows_count = len(lineups) * 8 + 1
    detail_ws = _get_or_create_worksheet(spreadsheet, tab("Detail"), detail_rows_count, 8)

    detail_header = ["Lineup #", "Player", "Salary", "Proj Pts", "Own%", "Value", "Std Dev", "Wave"]
    detail_rows = [detail_header]

    for i, lineup in enumerate(lineups, 1):
        total_sal = sum(p["salary"] for p in lineup)
        total_pts = sum(p["projected_points"] for p in lineup)
        for p in lineup:
            detail_rows.append([
                i,
                p["name"],
                p["salary"],
                p["projected_points"],
                p.get("proj_ownership", 0),
                p.get("value", 0),
                p.get("std_dev", 0),
                p.get("wave", "?"),
            ])
        detail_rows.append(["TOTAL", "", total_sal, round(total_pts, 1), "", "", "", ""])

    detail_ws.update(detail_rows, value_input_option="RAW")
    detail_ws.format("A1:H1", HEADER_FMT)

    # ── Tab 4: Contest Info (if available) ──
    if contest_profile and contest_metrics and contest_params:
        contest_ws = _get_or_create_worksheet(spreadsheet, tab("Contest"), 30, 4)

        contest_rows = [
            ["CONTEST DETAILS", "", "", ""],
            ["Name", contest_profile["name"], "", ""],
            ["Entry Fee", f"${contest_profile['entry_fee']:,}", "", ""],
            ["Prize Pool", f"${contest_profile['prize_pool']:,.0f}", "", ""],
            ["Field Size", f"{contest_profile['max_entries']:,}", "", ""],
            ["Entries", f"{contest_profile['entries']:,}", "", ""],
            ["Max Entries/User", contest_profile["max_entries_per_user"], "", ""],
            ["Payout Spots", f"{contest_profile['payout_spots']:,}", "", ""],
            ["1st Place", f"${contest_profile['first_place_prize']:,.0f}", "", ""],
            ["", "", "", ""],
            ["CLASSIFICATION", "", "", ""],
            ["Contest Type", contest_metrics["contest_type_display"], "", ""],
            ["Payout Skew", contest_metrics["payout_skew"], "", ""],
            ["Payout %", f"{contest_metrics['payout_pct']*100:.1f}%", "", ""],
            ["Top Heavy Ratio", f"{contest_metrics['top_heavy_ratio']*100:.1f}%", "", ""],
            ["Field Factor", contest_metrics["field_factor"], "", ""],
            ["", "", "", ""],
            ["OPTIMIZER PARAMETERS", "", "", ""],
            ["Lineups", contest_params["num_lineups"], "", ""],
            ["Kelly Fraction", contest_params["kelly_fraction"], "", ""],
            ["Lambda Base", contest_params["lambda_base"], "", ""],
            ["Lambda Penalty Cap", contest_params["lambda_penalty_cap"], "", ""],
            ["Simulations", contest_params["n_sims"], "", ""],
            ["Max Overlap (Early)", contest_params["max_overlap_early"], "", ""],
            ["Max Overlap (Late)", contest_params["max_overlap_late"], "", ""],
        ]

        contest_ws.update(contest_rows, value_input_option="RAW")

        for row_num in [1, 11, 18]:
            contest_ws.format(f"A{row_num}:D{row_num}", HEADER_FMT)

    # ── Tab 5: Sim Results (if available) ──
    if sim_results:
        sim_ws = _get_or_create_worksheet(spreadsheet, tab("Sim Results"), len(sim_results) + 5, 10)

        sim_rows = []

        # Summary row if available
        if sim_summary:
            sim_rows.append([
                "SIMULATION SUMMARY", "",
                f"Avg ROI: {sim_summary.get('avg_roi', 0):.1f}%", "",
                f"Best ROI: {sim_summary.get('best_roi', 0):.1f}%", "",
                f"Avg Cash Rate: {sim_summary.get('avg_cash_rate', 0):.1f}%",
                "", "", "",
            ])
            sim_rows.append([""] * 10)

        sim_rows.append([
            "Lineup #", "Players", "Salary", "Proj Pts",
            "Mean ROI %", "ROI Std", "Cash Rate %",
            "Mean Payout", "Avg Own %", "",
        ])

        for r in sim_results:
            sim_rows.append([
                r["lineup_id"],
                " | ".join(r["players"]),
                r["total_salary"],
                round(r["total_proj"], 1),
                round(r["mean_roi_pct"], 2),
                round(r["roi_std"], 2),
                round(r["cash_rate_pct"], 1),
                round(r["mean_payout"], 2),
                round(r["avg_ownership"], 1),
                "",
            ])

        sim_ws.update(sim_rows, value_input_option="RAW")

        # Format headers
        header_row = 3 if sim_summary else 1
        sim_ws.format(f"A{header_row}:J{header_row}", HEADER_FMT)
        if sim_summary:
            sim_ws.format("A1:J1", HEADER_FMT)

    # ── Tab 6: Portfolio Analytics (if available) ──
    if portfolio_analytics:
        exposure = portfolio_analytics.get("player_exposure", {})
        port_rows_count = max(len(exposure) + 20, 30)
        port_ws = _get_or_create_worksheet(spreadsheet, tab("Portfolio"), port_rows_count, 6)

        port_rows = []

        # Portfolio summary metrics
        port_rows.append(["PORTFOLIO SUMMARY", "", "", "", "", ""])
        port_rows.append(["", "", "", "", "", ""])
        port_rows.append(["Metric", "Value", "", "", "", ""])
        port_rows.append(["Portfolio Size", f"{portfolio_analytics.get('portfolio_size', 0)} lineups", "", "", "", ""])
        port_rows.append(["Total Cost", f"${portfolio_analytics.get('total_cost', 0):,.0f}", "", "", "", ""])
        port_rows.append(["Expected ROI", f"{portfolio_analytics.get('expected_roi_pct', 0):.2f}%", "", "", "", ""])
        port_rows.append(["Std Deviation", f"{portfolio_analytics.get('portfolio_std_pct', 0):.2f}%", "", "", "", ""])
        port_rows.append(["Sharpe Ratio", f"{portfolio_analytics.get('sharpe_ratio', 0):.4f}", "", "", "", ""])
        port_rows.append(["VaR (5%)", f"${portfolio_analytics.get('var_5_pct', 0):,.2f}", "", "", "", ""])
        port_rows.append(["VaR (1%)", f"${portfolio_analytics.get('var_1_pct', 0):,.2f}", "", "", "", ""])
        port_rows.append(["Avg Lineup Correlation", f"{portfolio_analytics.get('avg_lineup_correlation', 0):.4f}", "", "", "", ""])
        port_rows.append(["Diversification Ratio", f"{portfolio_analytics.get('diversification_ratio', 0):.4f}", "", "", "", ""])
        port_rows.append(["", "", "", "", "", ""])

        # Player exposure table
        port_rows.append(["PLAYER EXPOSURE", "", "", "", "", ""])
        port_rows.append(["Player", "Lineups", "Exposure %", "", "", ""])

        for name, data in exposure.items():
            port_rows.append([name, data["count"], f"{data['pct']:.1f}%", "", "", ""])

        port_ws.update(port_rows, value_input_option="RAW")
        port_ws.format("A1:F1", HEADER_FMT)
        port_ws.format("A3:F3", HEADER_FMT)
        port_ws.format(f"A{14}:F{14}", HEADER_FMT)
        port_ws.format(f"A{15}:F{15}", HEADER_FMT)

    # Clean up default Sheet1 if it exists
    try:
        default = spreadsheet.worksheet("Sheet1")
        if len(spreadsheet.worksheets()) > 1:
            spreadsheet.del_worksheet(default)
    except (gspread.WorksheetNotFound, gspread.exceptions.APIError):
        pass

    # Also clean up old un-prefixed tabs from previous runs
    old_tabs = ["Lineups", "Projections", "Lineup Detail", "Contest Info"]
    for old_name in old_tabs:
        try:
            old_ws = spreadsheet.worksheet(old_name)
            if len(spreadsheet.worksheets()) > 1:
                spreadsheet.del_worksheet(old_ws)
        except (gspread.WorksheetNotFound, gspread.exceptions.APIError):
            pass

    return spreadsheet.url
