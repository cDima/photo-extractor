#!/usr/bin/env python3
"""
Photo Year Travel Recap (macOS / Apple Photos)

- Reads your local Apple Photos library via osxphotos (no uploads).
- Builds a travel-focused year recap (filters your home metro area).
- Prints a friendly, shareable CLI summary.
- Writes JSON files next to this script for LLM consumption.

Run:
  python3 photo_year_recap.py 2025

Install:
  pip install osxphotos

Permissions:
  You may need to grant your terminal (iTerm/Terminal/VS Code) Photos and/or Full Disk Access.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

import osxphotos


# ----------------------------
# Simple “product” config (kept minimal)
# ----------------------------

HOME_RADIUS_KM = 30.0      # anything within this radius of home centroid is treated as home-region noise
TRAVEL_MIN_DAYS = 2        # a place needs >= this many distinct days to count as “worth remembering”
TOP_COUNTRIES = 6
TOP_CITIES = 10
TOP_WEEKS = 3
TOP_NEW_PLACES = 6

# How many example photo UUIDs to include per “story week” / place for finding photos later
EXAMPLE_UUIDS_PER_ITEM = 12


# ----------------------------
# Utilities
# ----------------------------

def safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_label(mk: str) -> str:
    _, m = mk.split("-")
    names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    return names[int(m) - 1]


def fmt_range(start: date, end: date) -> str:
    """Human-friendly range: Aug 4–10 or Jul 31–Aug 13."""
    if start.month == end.month:
        return f"{start.strftime('%b')} {start.day}\u2013{end.day}"
    if start.year == end.year:
        return f"{start.strftime('%b')} {start.day}\u2013{end.strftime('%b')} {end.day}"
    return f"{start.isoformat()}\u2013{end.isoformat()}"


def emoji_flag(country_code: Optional[str]) -> str:
    if not country_code:
        return ""
    cc = country_code.strip().upper()
    if len(cc) != 2 or not cc.isalpha():
        return ""
    return chr(ord(cc[0]) + 127397) + chr(ord(cc[1]) + 127397)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    x = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(x), sqrt(1 - x))


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ----------------------------
# Models
# ----------------------------

@dataclass(frozen=True)
class PlaceKey:
    city: str
    country: str
    country_code: Optional[str] = None

    def label(self) -> str:
        return f"{self.city}, {self.country}"


@dataclass
class PlaceAggregate:
    days: Set[date] = field(default_factory=set)
    months: Set[str] = field(default_factory=set)
    coords: List[Tuple[float, float]] = field(default_factory=list)
    first_day: Optional[date] = None
    last_day: Optional[date] = None
    uuids: List[str] = field(default_factory=list)

    def add(self, d: date, mk: str, latlon: Optional[Tuple[float, float]], uuid: Optional[str]) -> None:
        self.days.add(d)
        self.months.add(mk)
        if latlon:
            self.coords.append(latlon)
        if uuid and len(self.uuids) < 300:  # cap to avoid huge memory
            self.uuids.append(uuid)

        if self.first_day is None or d < self.first_day:
            self.first_day = d
        if self.last_day is None or d > self.last_day:
            self.last_day = d

    def day_count(self) -> int:
        return len(self.days)

    def month_count(self) -> int:
        return len(self.months)

    def centroid(self) -> Optional[Tuple[float, float]]:
        if not self.coords:
            return None
        lat = sum(c[0] for c in self.coords) / len(self.coords)
        lon = sum(c[1] for c in self.coords) / len(self.coords)
        return (lat, lon)


@dataclass
class WeekStory:
    iso_year: int
    iso_week: int
    start: date
    end: date
    route: List[PlaceKey] = field(default_factory=list)
    countries: Set[str] = field(default_factory=set)
    example_uuids: List[str] = field(default_factory=list)

    def range_str(self) -> str:
        return fmt_range(self.start, self.end)

    def chain_str(self) -> str:
        return " → ".join([p.city for p in self.route])


# ----------------------------
# Place extraction from Photos
# ----------------------------

def extract_place(photo) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (city, country, country_code), preferring photo.place PlaceInfo.
    PlaceInfo details vary by macOS version but commonly has:
      place.names.city (list), place.names.country (list), place.country_code, place.name
    """
    place = getattr(photo, "place", None)
    if place:
        try:
            names = getattr(place, "names", None)
            city = None
            country = None
            if names:
                city_list = getattr(names, "city", None)
                country_list = getattr(names, "country", None)
                if isinstance(city_list, list) and city_list:
                    city = safe_str(city_list[0])
                if isinstance(country_list, list) and country_list:
                    country = safe_str(country_list[0])

            if not city or not country:
                name = safe_str(getattr(place, "name", None))
                if name and "," in name:
                    parts = [p.strip() for p in name.split(",") if p.strip()]
                    if not country and parts:
                        country = safe_str(parts[-1])
                    if not city and len(parts) >= 2:
                        city = safe_str(parts[-2])

            country_code = safe_str(getattr(place, "country_code", None))
            return city, country, country_code
        except Exception:
            pass

    # fallback (rarely useful)
    return (
        safe_str(getattr(photo, "city", None)) if hasattr(photo, "city") else None,
        safe_str(getattr(photo, "country", None)) if hasattr(photo, "country") else None,
        safe_str(getattr(photo, "country_code", None)) if hasattr(photo, "country_code") else None,
    )


# ----------------------------
# Main pipeline
# ----------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 photo_year_recap.py <year>\nExample: python3 photo_year_recap.py 2025")
        raise SystemExit(2)

    try:
        year = int(sys.argv[1])
    except ValueError:
        print("Year must be an integer, e.g. 2025")
        raise SystemExit(2)

    db = osxphotos.PhotosDB()

    # Counters (for trust / diagnostics)
    photos_scanned = 0
    photos_in_year = 0
    photos_with_gps = 0
    photos_with_place = 0

    # Aggregates
    places: Dict[PlaceKey, PlaceAggregate] = defaultdict(PlaceAggregate)
    country_days_all: Dict[str, Set[date]] = defaultdict(set)

    # Store minimal per-photo row for travel rollups later
    # (d, mk, latlon, place_key, uuid)
    rows: List[Tuple[date, str, Optional[Tuple[float, float]], Optional[PlaceKey], str]] = []

    # Scan year
    for p in db.photos():
        photos_scanned += 1
        dt: Optional[datetime] = getattr(p, "date", None)
        if not dt or dt.year != year:
            continue
        photos_in_year += 1

        d = dt.date()
        mk = month_key(d)

        # UUID for later “find photos”
        uuid = safe_str(getattr(p, "uuid", None)) or ""

        # GPS
        latlon: Optional[Tuple[float, float]] = None
        loc = getattr(p, "location", None)
        if loc:
            try:
                lat, lon = loc
                if lat is not None and lon is not None:
                    latlon = (float(lat), float(lon))
                    photos_with_gps += 1
            except Exception:
                latlon = None

        # Place labels
        city, country, country_code = extract_place(p)
        city = safe_str(city)
        country = safe_str(country)
        country_code = safe_str(country_code)

        pk: Optional[PlaceKey] = None
        if city and country:
            photos_with_place += 1
            pk = PlaceKey(city=city, country=country, country_code=country_code)
            places[pk].add(d, mk, latlon, uuid)
            country_days_all[country].add(d)

        rows.append((d, mk, latlon, pk, uuid))

    if photos_in_year == 0:
        print(f"No photos found for {year}.")
        return

    # Infer home base: most distinct days among place-labeled photos
    home_base: Optional[PlaceKey] = None
    if places:
        home_base = max(places.items(), key=lambda kv: (kv[1].day_count(), kv[1].month_count()))[0]

    home_country = home_base.country if home_base else None
    home_center = places[home_base].centroid() if home_base else None

    def is_home_region(pk: PlaceKey) -> bool:
        if not home_center:
            return False
        c = places[pk].centroid()
        if not c:
            return False
        return haversine_km(home_center, c) <= HOME_RADIUS_KM

    # Travel places = outside home region AND >= TRAVEL_MIN_DAYS
    travel_places: Set[PlaceKey] = set()
    for pk, agg in places.items():
        if home_base and is_home_region(pk):
            continue
        if agg.day_count() < TRAVEL_MIN_DAYS:
            continue
        travel_places.add(pk)

    # Rollups: travel countries/cities/months/weeks
    travel_country_days: Dict[str, Set[date]] = defaultdict(set)
    travel_city_days: Dict[PlaceKey, Set[date]] = defaultdict(set)
    travel_month_cities: Dict[str, Set[PlaceKey]] = defaultdict(set)
    travel_month_countries: Dict[str, Set[str]] = defaultdict(set)
    travel_month_range: Dict[str, Tuple[Optional[date], Optional[date]]] = defaultdict(lambda: (None, None))

    # Week stories (travel-only)
    # key: (iso_year, iso_week) -> story
    week_routes: Dict[Tuple[int, int], List[PlaceKey]] = defaultdict(list)
    week_seen: Dict[Tuple[int, int], Set[PlaceKey]] = defaultdict(set)
    week_range: Dict[Tuple[int, int], Tuple[Optional[date], Optional[date]]] = defaultdict(lambda: (None, None))
    week_uuid_bucket: Dict[Tuple[int, int], List[str]] = defaultdict(list)

    for d, mk, latlon, pk, uuid in rows:
        if not pk or pk not in travel_places:
            continue

        travel_city_days[pk].add(d)
        travel_country_days[pk.country].add(d)

        travel_month_cities[mk].add(pk)
        travel_month_countries[mk].add(pk.country)
        a, b = travel_month_range[mk]
        if a is None or d < a:
            a = d
        if b is None or d > b:
            b = d
        travel_month_range[mk] = (a, b)

        iso_year, iso_week, _ = d.isocalendar()
        wk = (iso_year, iso_week)

        # week range
        wa, wb = week_range[wk]
        if wa is None or d < wa:
            wa = d
        if wb is None or d > wb:
            wb = d
        week_range[wk] = (wa, wb)

        # ordered route
        if pk not in week_seen[wk]:
            week_seen[wk].add(pk)
            week_routes[wk].append(pk)

        if uuid and len(week_uuid_bucket[wk]) < 500:
            week_uuid_bucket[wk].append(uuid)

    # Build “story-worthy” weeks
    week_stories: List[WeekStory] = []
    for (iso_y, iso_w), route in week_routes.items():
        if not route:
            continue
        wa, wb = week_range[(iso_y, iso_w)]
        if not wa or not wb:
            continue

        countries = {p.country for p in route}
        international = home_country is not None and any(c != home_country for c in countries)

        # only keep story weeks
        if len(route) >= 2 or len(countries) >= 2 or international:
            story = WeekStory(
                iso_year=iso_y,
                iso_week=iso_w,
                start=wa,
                end=wb,
                route=route,
                countries=countries,
                example_uuids=week_uuid_bucket[(iso_y, iso_w)][:EXAMPLE_UUIDS_PER_ITEM]
            )
            week_stories.append(story)

    week_stories.sort(key=lambda w: (-len(w.route), -len(w.countries), w.start))
    top_weeks = week_stories[:TOP_WEEKS]

    # Rankings
    countries_sorted = sorted(travel_country_days.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    cities_sorted = sorted(travel_city_days.items(), key=lambda kv: (-len(kv[1]), kv[0].label()))

    # Best month
    best_month: Optional[str] = None
    best_score = (-1, -1, "")
    for mk in sorted(travel_month_range.keys()):
        score = (len(travel_month_cities[mk]), len(travel_month_countries[mk]), mk)
        if score > best_score:
            best_score = score
            best_month = mk

    # New places: travel places visited in exactly one month
    new_places = [pk for pk in travel_places if places[pk].month_count() == 1]
    new_places.sort(key=lambda pk: (-places[pk].day_count(), pk.label()))
    new_places = new_places[:TOP_NEW_PLACES]

    # --------- Friendly CLI output (less geeky) ---------

    print(f"📍 Your {year} Travel Recap\n")

    # Quick punchline first
    print(f"You went beyond home to {len(travel_country_days)} countries and {len(travel_city_days)} cities.\n")

    # Most valuable: top countries + dates
    if countries_sorted:
        print("Top countries")
        for i, (country, days_set) in enumerate(countries_sorted[:TOP_COUNTRIES], 1):
            start, end = min(days_set), max(days_set)
            # try to find a country code from any city in that country
            cc = next((p.country_code for p in travel_places if p.country == country and p.country_code), None)
            flag = emoji_flag(cc)
            print(f"{i}. {flag} {country} — {len(days_set)} days ({fmt_range(start, end)})".strip())
        print()

    # Top cities (memorable)
    if cities_sorted:
        print("Top cities")
        for i, (pk, days_set) in enumerate(cities_sorted[:TOP_CITIES], 1):
            start, end = min(days_set), max(days_set)
            print(f"{i}. {pk.city}, {pk.country} — {len(days_set)} days ({fmt_range(start, end)})")
        print()

    # Best month
    if best_month:
        a, b = travel_month_range[best_month]
        rng = f"{fmt_range(a, b)}" if a and b else ""
        print("Most traveled month")
        print(f"• {month_label(best_month)} — {len(travel_month_cities[best_month])} cities, {len(travel_month_countries[best_month])} countries ({rng})\n")

    # Story weeks
    if top_weeks:
        print("Weeks that tell a story")
        for w in top_weeks:
            print(f"• {w.range_str()}: {w.chain_str()}")
        print()

    # New places
    if new_places:
        print("New places you discovered")
        for pk in new_places:
            agg = places[pk]
            if agg.first_day and agg.last_day:
                print(f"• {pk.city}, {pk.country} ({fmt_range(agg.first_day, agg.last_day)})")
            else:
                print(f"• {pk.city}, {pk.country}")
        print()

    # Subtle diagnostics at the end (not in your face)
    if home_base:
        hb_days = places[home_base].day_count()
        print(f"(FYI: home base detected as {home_base.city}; filtered nearby cities within {int(HOME_RADIUS_KM)} km.)")
    print("Tip: Use the date ranges above to jump to those photos in Apple Photos.\n")
    print("🎉 Happy New Year")

    # --------- JSON outputs for LLM / downstream ---------

    out_dir = script_dir()
    recap_path = out_dir / f"travel_recap_{year}.json"
    evidence_path = out_dir / f"travel_evidence_{year}.json"

    recap = {
        "year": year,
        "summary": {
            "travel_countries": len(travel_country_days),
            "travel_cities": len(travel_city_days),
            "home_base": home_base.city if home_base else None,
            "home_country": home_base.country if home_base else None,
            "home_radius_km": HOME_RADIUS_KM,
            "travel_min_days": TRAVEL_MIN_DAYS,
        },
        "top_countries": [
            {
                "country": country,
                "country_code": next((p.country_code for p in travel_places if p.country == country and p.country_code), None),
                "days": len(days_set),
                "date_range": {"start": min(days_set).isoformat(), "end": max(days_set).isoformat()},
            }
            for country, days_set in countries_sorted[:TOP_COUNTRIES]
        ],
        "top_cities": [
            {
                "city": pk.city,
                "country": pk.country,
                "country_code": pk.country_code,
                "days": len(days_set),
                "date_range": {"start": min(days_set).isoformat(), "end": max(days_set).isoformat()},
            }
            for pk, days_set in cities_sorted[:TOP_CITIES]
        ],
        "most_traveled_month": None if not best_month else {
            "month": best_month,
            "label": month_label(best_month),
            "cities": len(travel_month_cities[best_month]),
            "countries": len(travel_month_countries[best_month]),
            "date_range": {
                "start": travel_month_range[best_month][0].isoformat() if travel_month_range[best_month][0] else None,
                "end": travel_month_range[best_month][1].isoformat() if travel_month_range[best_month][1] else None,
            },
        },
        "story_weeks": [
            {
                "iso_year": w.iso_year,
                "iso_week": w.iso_week,
                "date_range": {"start": w.start.isoformat(), "end": w.end.isoformat()},
                "route": [{"city": p.city, "country": p.country, "country_code": p.country_code} for p in w.route],
            }
            for w in top_weeks
        ],
        "new_places": [
            {
                "city": pk.city,
                "country": pk.country,
                "country_code": pk.country_code,
                "date_range": {
                    "start": places[pk].first_day.isoformat() if places[pk].first_day else None,
                    "end": places[pk].last_day.isoformat() if places[pk].last_day else None,
                },
            }
            for pk in new_places
        ],
        "data_quality": {
            "photos_scanned": photos_scanned,
            "photos_in_year": photos_in_year,
            "photos_with_gps": photos_with_gps,
            "photos_with_place_names": photos_with_place,
        },
    }

    # Evidence file: attach example UUIDs so you can later build “open in Photos” workflows
    # (Apple Photos doesn’t provide a reliable built-in URL deep link to a specific asset. :contentReference[oaicite:2]{index=2})
    evidence = {
        "year": year,
        "notes": {
            "photos_deep_links": "Apple Photos has URL schemes to open the app (photos:// etc), but not a stable public deep link to open a specific asset by UUID. Consider using date+place search in Photos, or a third-party tool like Hookmark for per-photo links.",
        },
        "story_weeks": [
            {
                "date_range": {"start": w.start.isoformat(), "end": w.end.isoformat()},
                "route_text": w.chain_str(),
                "example_photo_uuids": w.example_uuids,
            }
            for w in top_weeks
        ],
        "places": [
            {
                "city": pk.city,
                "country": pk.country,
                "date_range": {
                    "start": places[pk].first_day.isoformat() if places[pk].first_day else None,
                    "end": places[pk].last_day.isoformat() if places[pk].last_day else None,
                },
                "example_photo_uuids": places[pk].uuids[:EXAMPLE_UUIDS_PER_ITEM],
            }
            for pk, _days_set in cities_sorted[:TOP_CITIES]
        ],
        "open_photos_app_urls": {
            "photos": "photos://",
            "photos_navigation": "photos-navigation://",
            "cloudphoto": "cloudphoto://",
        },
    }

    write_json(recap_path, recap)
    write_json(evidence_path, evidence)


if __name__ == "__main__":
    main()
