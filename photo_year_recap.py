#!/usr/bin/env python3
"""
photo_year_recap.py

A privacy-first Mac CLI that reads your local Apple Photos library (no uploads),
extracts geo + place labels, and prints a *travel-focused* year recap that filters
out your home metro noise (e.g., Woodinville/Seattle area).

USAGE
  python3 photo_year_recap.py 2025

DEPENDENCY
  pip install osxphotos

NOTES
- macOS will require permissions (Photos / Full Disk Access depending on setup).
- We prioritize Apple Photos' own PlaceInfo (reverse-geocoded names) when available.
- "Home region" is auto-inferred from your most-common city; nearby cities are filtered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

import osxphotos


# ----------------------------
# Config (kept intentionally simple / no CLI flags)
# ----------------------------

HOME_RADIUS_KM = 30.0          # anything within this radius of home centroid is treated as "home region"
TRAVEL_MIN_DAYS = 2            # ignore places that appear on fewer distinct days (filters drive-bys / airports)
TOP_N_COUNTRIES = 8
TOP_N_CITIES = 10
TOP_N_WEEKS = 3
TOP_N_NEW_PLACES = 6


# ----------------------------
# Utility helpers
# ----------------------------

def safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def fmt_range(start: date, end: date) -> str:
    """Human friendly day range like 'Aug 4–10' or 'Jul 31–Aug 13'."""
    if start.month == end.month:
        return f"{start.strftime('%b')} {start.day}\u2013{end.day}"
    if start.year == end.year:
        return f"{start.strftime('%b')} {start.day}\u2013{end.strftime('%b')} {end.day}"
    return f"{start.isoformat()}\u2013{end.isoformat()}"


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_label(mk: str) -> str:
    _, m = mk.split("-")
    names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    return names[int(m) - 1]


def emoji_flag(country_code: Optional[str]) -> str:
    """Convert ISO alpha-2 to emoji flag if possible."""
    if not country_code:
        return ""
    cc = country_code.strip().upper()
    if len(cc) != 2 or not cc.isalpha():
        return ""
    return chr(ord(cc[0]) + 127397) + chr(ord(cc[1]) + 127397)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distance between two lat/lon points in km."""
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    x = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(x), sqrt(1 - x))


# ----------------------------
# Domain models
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
    """Aggregates information for a place across photos."""
    days: Set[date] = field(default_factory=set)
    months: Set[str] = field(default_factory=set)
    coords: List[Tuple[float, float]] = field(default_factory=list)
    first_day: Optional[date] = None
    last_day: Optional[date] = None

    def add(self, d: date, mk: str, latlon: Optional[Tuple[float, float]]) -> None:
        self.days.add(d)
        self.months.add(mk)
        if latlon:
            self.coords.append(latlon)
        if self.first_day is None or d < self.first_day:
            self.first_day = d
        if self.last_day is None or d > self.last_day:
            self.last_day = d

    def day_count(self) -> int:
        return len(self.days)

    def month_count(self) -> int:
        return len(self.months)

    def date_range_str(self) -> str:
        if not self.first_day or not self.last_day:
            return ""
        return f"({fmt_range(self.first_day, self.last_day)})"

    def centroid(self) -> Optional[Tuple[float, float]]:
        if not self.coords:
            return None
        lat = sum(c[0] for c in self.coords) / len(self.coords)
        lon = sum(c[1] for c in self.coords) / len(self.coords)
        return (lat, lon)


@dataclass
class WeekAggregate:
    """Travel-story representation for a week."""
    start: Optional[date] = None
    end: Optional[date] = None
    route: List[PlaceKey] = field(default_factory=list)
    route_seen: Set[PlaceKey] = field(default_factory=set)
    countries: Set[str] = field(default_factory=set)

    def add_stop(self, d: date, place: PlaceKey) -> None:
        if self.start is None or d < self.start:
            self.start = d
        if self.end is None or d > self.end:
            self.end = d

        self.countries.add(place.country)

        # Keep an ordered route but de-dupe repeated visits within a week
        if place not in self.route_seen:
            self.route_seen.add(place)
            self.route.append(place)

    def range_str(self) -> str:
        if self.start and self.end:
            return fmt_range(self.start, self.end)
        return "Unknown week"

    def city_chain(self) -> str:
        return " → ".join([p.city for p in self.route])


# ----------------------------
# Place extraction (important)
# ----------------------------

def extract_place_from_photo(photo) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (city, country, country_code).

    Prefer PhotoInfo.place PlaceInfo (reverse geocoded by Photos).
    For PlaceInfo, we use place.names.city/country (often lists).
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

            # Fallback: parse the display name if needed
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

    # Final fallback (older versions / libraries): try simple fields if present
    city = safe_str(getattr(photo, "city", None)) if hasattr(photo, "city") else None
    country = safe_str(getattr(photo, "country", None)) if hasattr(photo, "country") else None
    country_code = safe_str(getattr(photo, "country_code", None)) if hasattr(photo, "country_code") else None
    return city, country, country_code


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 photo_year_recap.py <year>\nExample: python3 photo_year_recap.py 2025")
        raise SystemExit(2)

    try:
        target_year = int(sys.argv[1])
    except ValueError:
        print("Year must be an integer, e.g., 2025")
        raise SystemExit(2)

    # Open Photos DB (requires macOS permissions)
    db = osxphotos.PhotosDB()

    # Counters
    photos_scanned = 0
    photos_in_year = 0
    photos_with_gps = 0
    photos_with_place_names = 0

    # Aggregations
    all_places: Dict[PlaceKey, PlaceAggregate] = defaultdict(PlaceAggregate)
    all_countries: Dict[str, PlaceAggregate] = defaultdict(PlaceAggregate)

    # Travel-only aggregations (filled later)
    travel_country_days: Dict[str, Set[date]] = defaultdict(set)
    travel_city_days: Dict[PlaceKey, Set[date]] = defaultdict(set)

    # Month rollups (travel-only)
    travel_month_cities: Dict[str, Set[PlaceKey]] = defaultdict(set)
    travel_month_countries: Dict[str, Set[str]] = defaultdict(set)
    travel_month_range: Dict[str, Tuple[Optional[date], Optional[date]]] = defaultdict(lambda: (None, None))

    # Week rollups (travel-only)
    weeks: Dict[Tuple[int, int], WeekAggregate] = defaultdict(WeekAggregate)

    # Also track "raw" month ranges for the year (useful for diagnostics)
    year_month_range: Dict[str, Tuple[Optional[date], Optional[date]]] = defaultdict(lambda: (None, None))

    # We’ll store per-photo minimal info for travel classification and later week/month rollups
    PhotoRow = Tuple[date, str, Optional[Tuple[float, float]], Optional[PlaceKey]]
    photo_rows: List[PhotoRow] = []

    # -------- Pass 1: scan year + collect place aggregates --------
    for p in db.photos():
        photos_scanned += 1

        dt = getattr(p, "date", None)
        if not dt or dt.year != target_year:
            continue
        photos_in_year += 1

        d = dt.date()
        mk = month_key(d)

        # update year month range
        a, b = year_month_range[mk]
        if a is None or d < a:
            a = d
        if b is None or d > b:
            b = d
        year_month_range[mk] = (a, b)

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

        # Place names via Photos PlaceInfo
        city, country, country_code = extract_place_from_photo(p)
        city = safe_str(city)
        country = safe_str(country)
        country_code = safe_str(country_code)

        place_key: Optional[PlaceKey] = None
        if country and city:
            photos_with_place_names += 1
            place_key = PlaceKey(city=city, country=country, country_code=country_code)
            all_places[place_key].add(d, mk, latlon)
            all_countries[country].add(d, mk, latlon)

        photo_rows.append((d, mk, latlon, place_key))

    # If nothing in year, exit nicely
    if photos_in_year == 0:
        print(f"No photos found for year {target_year}.")
        return

    # -------- Infer home base and home region --------
    # Home base = place with most distinct days (requires place_key)
    home_base: Optional[PlaceKey] = None
    if all_places:
        home_base = max(all_places.items(), key=lambda kv: (kv[1].day_count(), kv[1].month_count()))[0]

    home_country: Optional[str] = home_base.country if home_base else None
    home_center: Optional[Tuple[float, float]] = all_places[home_base].centroid() if home_base else None

    def is_home_region(pk: PlaceKey) -> bool:
        """True if pk is within HOME_RADIUS_KM of home centroid."""
        if not home_center:
            return False
        c = all_places[pk].centroid()
        if not c:
            return False
        return haversine_km(home_center, c) <= HOME_RADIUS_KM

    # -------- Build travel sets (outside home region, min days) --------
    travel_places: Set[PlaceKey] = set()
    for pk, agg in all_places.items():
        if home_base and is_home_region(pk):
            continue
        if agg.day_count() < TRAVEL_MIN_DAYS:
            continue
        travel_places.add(pk)

    # If you have almost no travel places, still print something meaningful
    # (but likely you do, based on your output)

    # -------- Pass 2: populate travel rollups (countries, months, weeks) --------
    for d, mk, latlon, pk in photo_rows:
        if not pk or pk not in travel_places:
            continue

        travel_city_days[pk].add(d)
        travel_country_days[pk.country].add(d)

        travel_month_cities[mk].add(pk)
        travel_month_countries[mk].add(pk.country)

        # month range
        a, b = travel_month_range[mk]
        if a is None or d < a:
            a = d
        if b is None or d > b:
            b = d
        travel_month_range[mk] = (a, b)

        # week key: ISO year/week
        iso_year, iso_week, _ = d.isocalendar()
        wk = (iso_year, iso_week)
        weeks[wk].add_stop(d, pk)

    # -------- Derive sexy travel stats --------
    travel_countries_sorted = sorted(
        travel_country_days.items(),
        key=lambda kv: (-len(kv[1]), kv[0])
    )
    travel_cities_sorted = sorted(
        travel_city_days.items(),
        key=lambda kv: (-len(kv[1]), kv[0].label())
    )

    travel_country_count = len(travel_country_days)
    travel_city_count = len(travel_city_days)

    # Travel weeks: only weeks that have either multiple travel stops OR non-home country involved
    travel_weeks: List[WeekAggregate] = []
    for wk, agg in weeks.items():
        if not agg.route:
            continue
        # story-worthy if: 2+ cities OR 2+ countries OR any country != home_country
        international = home_country is not None and any(c != home_country for c in agg.countries)
        if len(agg.route) >= 2 or len(agg.countries) >= 2 or international:
            travel_weeks.append(agg)

    travel_weeks.sort(key=lambda a: (-len(a.route), -len(a.countries), a.start or date(target_year, 1, 1)))
    top_weeks = travel_weeks[:TOP_N_WEEKS]

    # Most traveled month: max distinct travel cities, tie-break by countries
    best_month: Optional[str] = None
    best_score = (-1, -1, "")
    for mk in sorted(travel_month_range.keys()):
        score = (len(travel_month_cities[mk]), len(travel_month_countries[mk]), mk)
        if score > best_score:
            best_score = score
            best_month = mk

    # New places: travel places visited in only one month, sorted by days
    new_places = [pk for pk in travel_places if all_places[pk].month_count() == 1]
    new_places.sort(key=lambda pk: (-all_places[pk].day_count(), pk.label()))
    new_places = new_places[:TOP_N_NEW_PLACES]

    # -------- Print output (travel-first, viral-friendly) --------
    print(f"📍 Where I Actually Traveled in {target_year}\n")

    # Diagnostics (trust-building)
    print("📸 Photo library access")
    print(f"• Photos scanned: {photos_scanned:,}")
    print(f"• Photos in {target_year}: {photos_in_year:,}")
    print(f"• Photos with GPS coordinates: {photos_with_gps:,}")
    print(f"• Photos with resolved place names: {photos_with_place_names:,}")

    # Home base info (small, but useful)
    if home_base:
        hb = all_places[home_base]
        hb_center = home_center
        hb_center_str = f"{hb_center[0]:.4f}, {hb_center[1]:.4f}" if hb_center else "unknown"
        print(f"• Home base inferred: {home_base.city} ({hb.day_count()} days) — filtering within {HOME_RADIUS_KM:.0f} km")
        print(f"  Home centroid: {hb_center_str}")
    else:
        print("• Home base inferred: (unknown)")
    print()

    # Travel totals
    travel_week_count = len({w.start.isocalendar()[:2] for w in travel_weeks if w.start}) if travel_weeks else 0
    print("You went beyond home:")
    print(f"• {travel_country_count} travel countries")
    print(f"• {travel_city_count} travel cities (≥ {TRAVEL_MIN_DAYS} days each)")
    print(f"• across {travel_week_count} travel weeks\n")

    # Countries
    print("━━━━━━━━━━━━━━━━━━━━━━")
    print("🌍 Countries worth remembering")
    if not travel_countries_sorted:
        print("No travel countries found (try lowering TRAVEL_MIN_DAYS in code).")
    else:
        for i, (country, days_set) in enumerate(travel_countries_sorted[:TOP_N_COUNTRIES], 1):
            # country_code is not always available at the country level; grab from any city in that country
            cc = None
            for pk in travel_places:
                if pk.country == country and pk.country_code:
                    cc = pk.country_code
                    break
            flag = emoji_flag(cc)
            # date range for travel days in that country (based on travel photo days)
            start = min(days_set)
            end = max(days_set)
            print(f"{i}. {flag} {country} — {len(days_set)} days ({fmt_range(start, end)})".strip())

    # Cities
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("🏙 Cities worth remembering")
    if not travel_cities_sorted:
        print("No travel cities found.")
    else:
        for i, (pk, days_set) in enumerate(travel_cities_sorted[:TOP_N_CITIES], 1):
            start = min(days_set)
            end = max(days_set)
            print(f"{i}. {pk.city}, {pk.country} — {len(days_set)} days ({fmt_range(start, end)})")

    # Month
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("📆 Your most traveled month")
    if best_month:
        a, b = travel_month_range[best_month]
        rng = f"({fmt_range(a, b)})" if a and b else ""
        print(f"• {month_label(best_month)} — {len(travel_month_cities[best_month])} cities, {len(travel_month_countries[best_month])} countries")
        if rng:
            print(f"  {rng}")
    else:
        print("• (No travel month found)")

    # Weeks
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("🧳 Weeks that tell a story")
    if not top_weeks:
        print("• (No story-worthy travel weeks found)")
    else:
        for w in top_weeks:
            print(f"• {w.range_str()}")
            print(f"  {w.city_chain()}")

    # New places
    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("✨ New places you discovered")
    if not new_places:
        print("• (No new travel places found)")
    else:
        for pk in new_places:
            agg = all_places[pk]
            # show the date range for that place
            if agg.first_day and agg.last_day:
                print(f"• {pk.city}, {pk.country} ({fmt_range(agg.first_day, agg.last_day)})")
            else:
                print(f"• {pk.city}, {pk.country}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("🎉 Happy New Year")
    print("Tip: Use the date ranges above to jump straight to those photos in Apple Photos.")


if __name__ == "__main__":
    import sys
    main()
