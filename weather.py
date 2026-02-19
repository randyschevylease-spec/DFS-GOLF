"""Weather & Tee-Time Wave Adjustment Module

Fetches hourly weather forecasts via Open-Meteo (free, no API key needed)
and calculates the AM vs PM wave advantage for DFS golf projection adjustments.

Uses the DataGolf 'am' field (1=AM wave Thursday, 0=PM wave Thursday).
Waves flip on Friday (AM→PM, PM→AM), so both days are factored in.
"""

import requests
from datetime import date, timedelta

# ── Course Location Database ─────────────────────────────────────────────────
# Maps DataGolf event names (lowercased substring match) to (lat, lon, timezone)
# Add new events as needed — coordinates are for the course, not the city.

COURSE_LOCATIONS = {
    "genesis invitational": (34.0489, -118.5005, "America/Los_Angeles"),        # Riviera CC, Pacific Palisades
    "farmers insurance": (32.8997, -117.2524, "America/Los_Angeles"),           # Torrey Pines, La Jolla
    "phoenix open": (33.6420, -111.9085, "America/Phoenix"),                    # TPC Scottsdale
    "at&t pebble beach": (36.5685, -121.9500, "America/Los_Angeles"),           # Pebble Beach GL
    "pebble beach": (36.5685, -121.9500, "America/Los_Angeles"),
    "players championship": (30.1975, -81.3942, "America/New_York"),            # TPC Sawgrass
    "the players": (30.1975, -81.3942, "America/New_York"),
    "arnold palmer": (28.4602, -81.5068, "America/New_York"),                   # Bay Hill, Orlando
    "bay hill": (28.4602, -81.5068, "America/New_York"),
    "masters": (33.5030, -82.0230, "America/New_York"),                         # Augusta National
    "rbc heritage": (32.1330, -80.8025, "America/New_York"),                    # Harbour Town, Hilton Head
    "harbour town": (32.1330, -80.8025, "America/New_York"),
    "zurich classic": (29.8886, -90.0868, "America/Chicago"),                   # TPC Louisiana
    "wells fargo": (35.0535, -80.8510, "America/New_York"),                     # Quail Hollow, Charlotte
    "quail hollow": (35.0535, -80.8510, "America/New_York"),
    "pga championship": (None, None, None),                                     # Rotates — set manually
    "charles schwab": (32.9334, -97.0975, "America/Chicago"),                   # Colonial CC, Fort Worth
    "colonial": (32.9334, -97.0975, "America/Chicago"),
    "memorial": (40.0538, -83.1731, "America/New_York"),                        # Muirfield Village, Dublin OH
    "us open": (None, None, None),                                              # Rotates
    "u.s. open": (None, None, None),
    "travelers": (41.7381, -72.7401, "America/New_York"),                       # TPC River Highlands, Cromwell CT
    "john deere": (41.4672, -90.4280, "America/Chicago"),                       # TPC Deere Run
    "scottish open": (56.0600, -2.7800, "Europe/London"),                       # Varies, Scotland
    "the open": (None, None, None),                                             # Rotates
    "open championship": (None, None, None),
    "rocket mortgage": (42.3586, -83.2110, "America/New_York"),                 # Detroit GC
    "3m open": (44.8520, -93.4451, "America/Chicago"),                          # TPC Twin Cities
    "wyndham": (36.0560, -79.8300, "America/New_York"),                         # Sedgefield CC, Greensboro
    "fedex st. jude": (35.0658, -89.7700, "America/Chicago"),                   # TPC Southwind, Memphis
    "bmw championship": (None, None, None),                                     # Rotates
    "tour championship": (33.8570, -84.3570, "America/New_York"),               # East Lake, Atlanta
    "rbc canadian": (43.8400, -79.3300, "America/New_York"),                    # Various, Canada
    "sony open": (21.2879, -157.7936, "Pacific/Honolulu"),                      # Waialae CC, Honolulu
    "sentry": (20.9200, -156.6800, "Pacific/Honolulu"),                         # Kapalua, Maui
    "amex": (33.7169, -116.2386, "America/Los_Angeles"),                        # La Quinta/PGA West
    "american express": (33.7169, -116.2386, "America/Los_Angeles"),
    "cognizant classic": (26.8371, -80.0893, "America/New_York"),               # PGA National, Palm Beach Gardens
    "honda classic": (26.8371, -80.0893, "America/New_York"),
    "valspar": (28.0520, -82.7260, "America/New_York"),                         # Innisbrook, Palm Harbor FL
    "rbc heritage": (32.1330, -80.8025, "America/New_York"),
    "byron nelson": (33.0900, -96.8200, "America/Chicago"),                     # TPC Craig Ranch, McKinney TX
    "dean & deluca": (32.9334, -97.0975, "America/Chicago"),                    # Colonial
    "cincinnati": (39.1950, -84.4240, "America/New_York"),                      # TPC River's Bend
    "fortinet": (38.4610, -122.7130, "America/Los_Angeles"),                    # Silverado, Napa
    "shriners": (36.1540, -115.2590, "America/Los_Angeles"),                    # TPC Summerlin, Las Vegas
    "cj cup": (36.1540, -115.2590, "America/Los_Angeles"),                      # Las Vegas area
    "zozo": (36.1540, -115.2590, "America/Los_Angeles"),                        # Varies
    "sanderson farms": (32.3520, -90.0240, "America/Chicago"),                  # CC of Jackson, MS
    "houston open": (29.9590, -95.6000, "America/Chicago"),                     # Memorial Park, Houston
    "rsm classic": (31.1580, -81.3810, "America/New_York"),                     # Sea Island, GA
}


def find_course_location(event_name):
    """Match an event name to course coordinates.

    Returns (lat, lon, timezone) or (None, None, None) if not found.
    """
    event_lower = event_name.lower()
    for key, loc in COURSE_LOCATIONS.items():
        if key in event_lower:
            return loc
    return (None, None, None)


def get_forecast(lat, lon, timezone, tournament_dates):
    """Fetch hourly weather forecast from Open-Meteo.

    tournament_dates: list of date strings ['2026-02-20', '2026-02-21']

    Returns dict with hourly data: time, wind_speed_10m, precipitation_probability,
    temperature_2m, wind_gusts_10m.
    """
    start_date = min(tournament_dates)
    end_date = max(tournament_dates)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_gusts_10m,precipitation_probability,temperature_2m",
        "wind_speed_unit": "mph",
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_wave_conditions(forecast_data, tournament_dates):
    """Parse hourly forecast into AM wave vs PM wave conditions.

    AM wave plays approximately hours 7-14 (7am-2pm).
    PM wave plays approximately hours 11-19 (11am-7pm).

    Returns dict with per-day AM and PM conditions.
    """
    hourly = forecast_data.get("hourly", {})
    times = hourly.get("time", [])
    winds = hourly.get("wind_speed_10m", [])
    gusts = hourly.get("wind_gusts_10m", [])
    precip = hourly.get("precipitation_probability", [])
    temps = hourly.get("temperature_2m", [])

    # Build hourly lookup: {date: {hour: {wind, gust, precip, temp}}}
    hourly_data = {}
    for i, t in enumerate(times):
        # Format: "2026-02-20T07:00"
        d = t[:10]
        h = int(t[11:13])
        if d not in hourly_data:
            hourly_data[d] = {}
        hourly_data[d][h] = {
            "wind": winds[i] if i < len(winds) else 0,
            "gust": gusts[i] if i < len(gusts) else 0,
            "precip": precip[i] if i < len(precip) else 0,
            "temp": temps[i] if i < len(temps) else 60,
        }

    wave_conditions = {}
    for d in tournament_dates:
        if d not in hourly_data:
            continue

        day_data = hourly_data[d]

        # AM wave: hours 7-14
        am_hours = [day_data[h] for h in range(7, 15) if h in day_data]
        # PM wave: hours 11-19
        pm_hours = [day_data[h] for h in range(11, 20) if h in day_data]

        def avg_conditions(hours):
            if not hours:
                return {"wind": 0, "gust": 0, "precip": 0, "temp": 60}
            return {
                "wind": sum(h["wind"] for h in hours) / len(hours),
                "gust": max(h["gust"] for h in hours),
                "precip": sum(h["precip"] for h in hours) / len(hours),
                "temp": sum(h["temp"] for h in hours) / len(hours),
            }

        wave_conditions[d] = {
            "am": avg_conditions(am_hours),
            "pm": avg_conditions(pm_hours),
        }

    return wave_conditions


def calculate_wave_adjustment(wave_conditions, tournament_dates):
    """Calculate the fantasy point adjustment for AM vs PM wave.

    Positive value = AM wave has the advantage (calmer/drier).
    Negative value = PM wave has the advantage.

    Players in the advantaged wave get a boost; disadvantaged wave gets a penalty.
    The adjustment is split: +half for good wave, -half for bad wave.

    Key factors:
    - Wind speed differential: each 5 mph difference ≈ 1.5 DK pts
    - Rain probability: 50%+ rain gap ≈ 2-4 DK pts
    - Gusts: sustained gusts >25 mph amplify the penalty

    Thursday: AM-wave players play AM, PM-wave players play PM
    Friday: waves flip — AM-wave players play PM, PM-wave players play AM
    """
    if not wave_conditions or not tournament_dates:
        return 0.0, {}

    # Thursday (day 1): AM=am_wave players, PM=pm_wave players
    # Friday (day 2): AM=pm_wave players, PM=am_wave players (waves flip)
    thursday = tournament_dates[0]
    friday = tournament_dates[1] if len(tournament_dates) > 1 else None

    total_am_advantage = 0.0  # Positive = AM wave is better on Thursday
    details = {}

    # Thursday conditions
    if thursday in wave_conditions:
        tc = wave_conditions[thursday]
        am_wind = tc["am"]["wind"]
        pm_wind = tc["pm"]["wind"]
        am_precip = tc["am"]["precip"]
        pm_precip = tc["pm"]["precip"]
        am_gust = tc["am"]["gust"]
        pm_gust = tc["pm"]["gust"]

        # Wind advantage (pts per mph differential, scaled by 0.3)
        wind_diff = pm_wind - am_wind  # Positive = PM is windier = AM advantage
        wind_adj_thu = wind_diff * 0.30

        # Gust amplifier: if gusts >25 mph in one wave, extra penalty
        gust_adj_thu = 0.0
        if pm_gust > 25 and am_gust < 20:
            gust_adj_thu = 1.5  # AM advantage
        elif am_gust > 25 and pm_gust < 20:
            gust_adj_thu = -1.5  # PM advantage

        # Rain advantage (precip prob is 0-100)
        rain_diff = (pm_precip - am_precip) / 100.0  # Positive = PM is wetter = AM advantage
        rain_adj_thu = rain_diff * 4.0

        thu_total = wind_adj_thu + gust_adj_thu + rain_adj_thu
        details["thursday"] = {
            "am_wind": am_wind, "pm_wind": pm_wind,
            "am_precip": am_precip, "pm_precip": pm_precip,
            "am_gust": am_gust, "pm_gust": pm_gust,
            "wind_adj": wind_adj_thu, "gust_adj": gust_adj_thu,
            "rain_adj": rain_adj_thu, "total": thu_total,
        }
        total_am_advantage += thu_total

    # Friday conditions (waves flip: AM-wave players now play PM)
    if friday and friday in wave_conditions:
        fc = wave_conditions[friday]
        am_wind = fc["am"]["wind"]
        pm_wind = fc["pm"]["wind"]
        am_precip = fc["am"]["precip"]
        pm_precip = fc["pm"]["precip"]
        am_gust = fc["am"]["gust"]
        pm_gust = fc["pm"]["gust"]

        wind_diff = am_wind - pm_wind  # Flipped! AM-wave players play PM Friday
        wind_adj_fri = wind_diff * 0.30

        gust_adj_fri = 0.0
        if am_gust > 25 and pm_gust < 20:
            gust_adj_fri = 1.5  # AM-wave (playing PM Fri) has advantage
        elif pm_gust > 25 and am_gust < 20:
            gust_adj_fri = -1.5

        rain_diff = (am_precip - pm_precip) / 100.0  # Flipped
        rain_adj_fri = rain_diff * 4.0

        fri_total = wind_adj_fri + gust_adj_fri + rain_adj_fri
        details["friday"] = {
            "am_wind": am_wind, "pm_wind": pm_wind,
            "am_precip": am_precip, "pm_precip": pm_precip,
            "am_gust": am_gust, "pm_gust": pm_gust,
            "wind_adj": wind_adj_fri, "gust_adj": gust_adj_fri,
            "rain_adj": rain_adj_fri, "total": fri_total,
        }
        total_am_advantage += fri_total

    # Cap the total adjustment at ±8 pts (avoid over-adjusting)
    total_am_advantage = max(-8.0, min(8.0, total_am_advantage))

    return total_am_advantage, details


def get_tournament_dates():
    """Determine the upcoming Thursday and Friday dates for the tournament.

    If today is Mon-Wed, use this week's Thu/Fri.
    If today is Thu-Sun, use this week's Thu/Fri (tournament in progress).
    """
    today = date.today()
    weekday = today.weekday()  # 0=Monday, 3=Thursday

    if weekday <= 3:  # Mon-Thu: this week's tournament
        days_until_thu = 3 - weekday
    else:  # Fri-Sun: this week's tournament already started
        days_until_thu = weekday - 3
        days_until_thu = -(weekday - 3)  # Go back to Thursday

    thursday = today + timedelta(days=(3 - weekday) if weekday <= 3 else -(weekday - 3))
    friday = thursday + timedelta(days=1)

    return [thursday.isoformat(), friday.isoformat()]


def get_wave_adjustment(event_name):
    """Main entry point: get the AM-wave advantage for a tournament.

    Returns:
        am_advantage: float — positive means AM-wave players get a boost,
                       negative means PM-wave players get a boost.
                       Each player gets +/- (am_advantage / 2).
        details: dict with per-day breakdown for display.
        wave_conditions: dict with raw weather data per wave.
    """
    lat, lon, tz = find_course_location(event_name)
    if lat is None:
        print(f"  Weather: No course location found for '{event_name}'")
        print(f"  Add it to COURSE_LOCATIONS in weather.py to enable wave adjustments.")
        return 0.0, {}, {}

    tournament_dates = get_tournament_dates()
    print(f"  Course: {event_name}")
    print(f"  Location: ({lat:.4f}, {lon:.4f}), TZ: {tz}")
    print(f"  Dates: R1={tournament_dates[0]}, R2={tournament_dates[1]}")

    try:
        forecast = get_forecast(lat, lon, tz, tournament_dates)
    except Exception as e:
        print(f"  Weather API error: {e}")
        return 0.0, {}, {}

    wave_conditions = parse_wave_conditions(forecast, tournament_dates)

    am_advantage, details = calculate_wave_adjustment(wave_conditions, tournament_dates)

    # Print weather summary
    for day_key in ["thursday", "friday"]:
        day_label = day_key.capitalize()
        d = tournament_dates[0] if day_key == "thursday" else tournament_dates[1] if len(tournament_dates) > 1 else None
        if d and d in wave_conditions:
            wc = wave_conditions[d]
            print(f"\n  {day_label} ({d}):")
            print(f"    AM wave: wind {wc['am']['wind']:.1f} mph (gust {wc['am']['gust']:.0f}), "
                  f"rain {wc['am']['precip']:.0f}%, temp {wc['am']['temp']:.0f}F")
            print(f"    PM wave: wind {wc['pm']['wind']:.1f} mph (gust {wc['pm']['gust']:.0f}), "
                  f"rain {wc['pm']['precip']:.0f}%, temp {wc['pm']['temp']:.0f}F")
            if day_key in details:
                dd = details[day_key]
                print(f"    Adj: wind {dd['wind_adj']:+.1f}, gust {dd['gust_adj']:+.1f}, "
                      f"rain {dd['rain_adj']:+.1f} → {dd['total']:+.1f} pts")

    if abs(am_advantage) < 0.5:
        verdict = "NEUTRAL — minimal wave advantage"
    elif am_advantage > 0:
        verdict = f"AM WAVE +{am_advantage/2:.1f} pts | PM WAVE {-am_advantage/2:.1f} pts"
    else:
        verdict = f"PM WAVE +{-am_advantage/2:.1f} pts | AM WAVE {am_advantage/2:.1f} pts"

    print(f"\n  Wave verdict: {verdict}")

    return am_advantage, details, wave_conditions
