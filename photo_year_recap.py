#!/usr/bin/env python3
"""
photo_year_recap.py

Mac-only CLI: reads your Apple Photos library (locally) and prints a human, viral,
photo-finding-friendly travel recap for a given year (e.g., 2025).

Setup (once):
  python3 -m pip install --user osxphotos

Run:
  python3 photo_year_recap.py 2025

Notes:
- This script DOES NOT upload anything. It reads your local Photos library database.
- Location names (city/country) come from what Photos already knows. If a photo only has
  lat/lon but Photos hasn't resolved it to a city/country, that photo may be omitted
  from city-based stats (still counted in "photos with location" if lat/lon exists).
"""

from __future__ import annotations

import sys
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import osxphotos


HOME_RADIUS_KM = 30.0


# --------- small helpers ---------

def _safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def _get_first_attr(obj, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            # osxphotos sometimes stores as properties
            try:
                v = v() if callable(v) else v
            except Exception:
                pass
            v = _safe_str(v)
            if v:
                return v
    return None

def _fmt_range(start: date, end: date) -> str:
    # "Aug 10–14" or "Jan 1–Dec 31"
    if start.month == end.month:
        return f"{start.strftime('%b')} {start.day}\u2013{end.day}"
    if start.year == end.year:
        return f"{start.strftime('%b')} {start.day}\u2013{end.strftime('%b')} {end.day}"
    return f"{start.isoformat()}\u2013{end.isoformat()}"

def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"

def _month_label(mk: str) -> str:
    # "2025-08" -> "August"
    y, m = mk.split("-")
    month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    return month_names[int(m)-1]

def _iso_week_label(week_key: Tuple[int, int, int]) -> str:
    # (iso_year, iso_week, iso_weekday_dummy)
    iso_year, iso_week, _ = week_key
    return f"{iso_year}-W{iso_week:02d}"

def _emoji_flag(country_code: Optional[str]) -> str:
    # Minimal: if we have ISO-3166 alpha2 code, render emoji flag.
    if not country_code or len(country_code) != 2:
        return ""
    cc = country_code.upper()
    if not cc.isalpha():
        return ""
    return chr(ord(cc[0]) + 127397) + chr(ord(cc[1]) + 127397)


# --------- core data ---------

@dataclass(frozen=True)
class PlaceKey:
    city: str
    country: str

    def city_label(self) -> str:
        return f"{self.city}, {self.country}"

@dataclass
class PlaceStats:
    days: set  # set[date]
    first_day: Optional[date] = None
    last_day: Optional[date] = None

    def add_day(self, d: date):
        self.days.add(d)
        if self.first_day is None or d < self.first_day:
            self.first_day = d
        if self.last_day is None or d > self.last_day:
            self.last_day = d

    def day_count(self) -> int:
        return len(self.days)

    def date_range_str(self) -> str:
        if not self.first_day or not self.last_day:
            return ""
        return f"({_fmt_range(self.first_day, self.last_day)})"

@dataclass
class CountryKey:
    country: str
    code: Optional[str] = None

    def label(self) -> str:
        flag = _emoji_flag(self.code)
        return f"{flag} {self.country}".strip()
    

def haversine_km(a, b):
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    x = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(x), sqrt(1-x))

def _extract_place(photo) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (city, country, country_code).

    Prefer PhotoInfo.place (PlaceInfo object) when available because it contains
    Apple's reverse geolocation info (city/country/etc). Fallback to older/other
    fields if present.
    """
    place = getattr(photo, "place", None)

    # 1) Best case: PlaceInfo object (Photos has reverse-geocoded names)
    # PlaceInfo has .names (PlaceNames) and .country_code, and .name string. :contentReference[oaicite:1]{index=1}
    if place:
        try:
            # Some versions: place.names.city is a list; Photos display uses first item
            names = getattr(place, "names", None)
            city = None
            country = None

            if names:
                city_list = getattr(names, "city", None)
                country_list = getattr(names, "country", None)

                if isinstance(city_list, list) and city_list:
                    city = _safe_str(city_list[0])
                if isinstance(country_list, list) and country_list:
                    country = _safe_str(country_list[0])

            # Fallback: if city missing, try parsing place.name (string Photos shows)
            # Example: "Mayfair, ... , Victoria, British Columbia, Canada" :contentReference[oaicite:2]{index=2}
            if not city or not country:
                name = _safe_str(getattr(place, "name", None))
                if name and "," in name:
                    parts = [p.strip() for p in name.split(",") if p.strip()]
                    # heuristic: last token is country, first token is "place name"
                    if not country and parts:
                        country = _safe_str(parts[-1])
                    if not city and len(parts) >= 2:
                        # often second-to-last is city-ish; if not, still better than nothing
                        city = _safe_str(parts[-2])

            country_code = _safe_str(getattr(place, "country_code", None))
            if city or country:
                return city, country, country_code
        except Exception:
            # If PlaceInfo shape differs, fall through
            pass

    # 2) Fallback: try assorted PhotoInfo fields (varies by version/library)
    city = _get_first_attr(photo, [
        "city", "place_city", "location_city", "sub_locality", "locality"
    ])
    country = _get_first_attr(photo, [
        "country", "place_country", "location_country"
    ])
    country_code = _get_first_attr(photo, [
        "country_code", "place_country_code", "location_country_code"
    ])

    # 3) Last resort: use any place-like string field as "city"
    if not city:
        maybe_place = _get_first_attr(photo, ["place_name", "location_name", "placename"])
        if maybe_place and country and maybe_place.lower() != country.lower():
            city = maybe_place

    return city, country, country_code



def main():
    if len(sys.argv) != 2:
        print("Usage: python3 photo_year_recap.py <year>\nExample: python3 photo_year_recap.py 2025")
        sys.exit(2)

    photos_with_place_names = 0

    try:
        target_year = int(sys.argv[1])
    except ValueError:
        print("Year must be an integer, e.g., 2025")
        sys.exit(2)

    # Open Photos DB (default system library)
    db = osxphotos.PhotosDB()

    photos_scanned = 0
    photos_with_location = 0

    # Stats structures
    city_stats: Dict[PlaceKey, PlaceStats] = defaultdict(lambda: PlaceStats(days=set()))
    country_stats: Dict[str, PlaceStats] = defaultdict(lambda: PlaceStats(days=set()))
    country_code_by_name: Dict[str, str] = {}

    # For week routes: track first-seen order per week
    # week_key = (iso_year, iso_week, 0)
    week_places_order: Dict[Tuple[int, int, int], List[PlaceKey]] = defaultdict(list)
    week_places_seen: Dict[Tuple[int, int, int], set] = defaultdict(set)
    week_first_day: Dict[Tuple[int, int, int], date] = {}
    week_last_day: Dict[Tuple[int, int, int], date] = {}

    # For month travel intensity
    month_cities: Dict[str, set] = defaultdict(set)
    month_countries: Dict[str, set] = defaultdict(set)
    month_day_ranges: Dict[str, Tuple[Optional[date], Optional[date]]] = defaultdict(lambda: (None, None))

    # Count city appearances by month to detect repeats
    city_months: Dict[PlaceKey, set] = defaultdict(set)

    # Iterate
    for p in db.photos():
        photos_scanned += 1
        dt = getattr(p, "date", None)
        if not dt:
            continue
        if dt.year != target_year:
            continue

        # Location presence (lat/lon)
        loc = getattr(p, "location", None)
        has_latlon = False
        if loc:
            try:
                lat, lon = loc
                has_latlon = (lat is not None and lon is not None)
            except Exception:
                has_latlon = False
        if has_latlon:
            photos_with_location += 1

        d = dt.date()
        mk = _month_key(d)
        iso_year, iso_week, _ = d.isocalendar()
        wk = (iso_year, iso_week, 0)

        # Update month day range
        mstart, mend = month_day_ranges[mk]
        if mstart is None or d < mstart:
            mstart = d
        if mend is None or d > mend:
            mend = d
        month_day_ranges[mk] = (mstart, mend)

        # Extract place names
        city, country, country_code = _extract_place(p)
        city = _safe_str(city)
        country = _safe_str(country)

        if city or country:
            photos_with_place_names += 1

        if country and country_code and country not in country_code_by_name:
            country_code_by_name[country] = country_code

        # If Photos doesn't provide city/country labels, we can't include it in city/country ranking.
        if not country:
            continue

        # Country stats
        country_stats[country].add_day(d)
        month_countries[mk].add(country)

        if city:
            pk = PlaceKey(city=city, country=country)
            city_stats[pk].add_day(d)
            month_cities[mk].add(pk)
            city_months[pk].add(mk)

            # Week routes (ordered)
            if pk not in week_places_seen[wk]:
                week_places_seen[wk].add(pk)
                week_places_order[wk].append(pk)

            # Week day range
            if wk not in week_first_day or d < week_first_day[wk]:
                week_first_day[wk] = d
            if wk not in week_last_day or d > week_last_day[wk]:
                week_last_day[wk] = d

    # Derived totals
    unique_countries = len(country_stats)
    unique_cities = len(city_stats)

    # "Distinct weeks traveled": weeks with >=1 known city
    traveled_weeks = [wk for wk, places in week_places_order.items() if places]
    distinct_weeks_traveled = len(traveled_weeks)

    # Rank top countries/cities by distinct days
    top_countries = sorted(country_stats.items(), key=lambda kv: (-kv[1].day_count(), kv[0]))[:10]
    top_cities = sorted(city_stats.items(), key=lambda kv: (-kv[1].day_count(), kv[0].city_label()))[:10]

    # Determine home base: city present in >= 6 distinct months, top by days
    home_base: Optional[PlaceKey] = None
    home_candidates = [(pk, city_stats[pk].day_count()) for pk in city_stats if len(city_months[pk]) >= 6]
    if home_candidates:
        home_candidates.sort(key=lambda x: (-x[1], x[0].city_label()))
        home_base = home_candidates[0][0]

    # Month traveled the most: max distinct cities, tie-break by countries
    best_month = None
    best_month_score = (-1, -1, "")
    for mk in sorted(month_day_ranges.keys()):
        cities_count = len(month_cities[mk])
        countries_count = len(month_countries[mk])
        score = (cities_count, countries_count, mk)
        if score > best_month_score:
            best_month_score = score
            best_month = mk

    # Most traveled weeks: weeks with >=2 distinct cities (or >=2 countries via their cities)
    week_scores = []
    for wk in traveled_weeks:
        places = week_places_order[wk]
        if not places:
            continue
        countries_in_week = {pk.country for pk in places}
        score = (len(places), len(countries_in_week))
        if score[0] >= 2 or score[1] >= 2:
            week_scores.append((score, wk))

    # Sort: most cities, then most countries, then earliest week
    week_scores.sort(key=lambda x: (-x[0][0], -x[0][1], x[1][0], x[1][1]))
    top_weeks = [wk for _, wk in week_scores[:3]]

    # New places discovered: cities visited in only one month; pick top 5 by days
    new_places = [(pk, city_stats[pk].day_count()) for pk in city_stats if len(city_months[pk]) == 1]
    new_places.sort(key=lambda x: (-x[1], x[0].city_label()))
    new_places = [pk for pk, _ in new_places[:5]]

    # Repeat places: visited in >=3 distinct months; pick top 5 by days (excluding home base)
    repeat_places = [(pk, city_stats[pk].day_count()) for pk in city_stats if len(city_months[pk]) >= 3 and pk != home_base]
    repeat_places.sort(key=lambda x: (-x[1], x[0].city_label()))
    repeat_places = [pk for pk, _ in repeat_places[:5]]

    # Print output
    print(f"📍 Where I Was in {target_year}\n")

    print("📸 Photo library access")
    print(f"• Photos scanned: {photos_scanned:,}")
    print(f"• Photos with GPS coordinates: {photos_with_location:,}")
    print(f"• Photos with resolved place names: {photos_with_place_names:,}")

    if photos_with_location > 0 and photos_with_place_names == 0:
        print("\n⚠️ Place names not available")
        print("Apple Photos has GPS data, but city/country names are not available.")
        print("Open Photos → Places and allow it to finish processing,")
        print("or use reverse geocoding fallback.\n")
    else:
        print()

    print("You traveled to:")
    print(f"• {unique_countries} countries")
    print(f"• {unique_cities} cities")
    print(f"• across {distinct_weeks_traveled} distinct weeks\n")

    print("━━━━━━━━━━━━━━━━━━━━━━")
    print("🌍 Top destination countries")
    if not top_countries:
        print("No country data found (Photos may not have resolved locations to country/city).")
    else:
        for i, (country, stats) in enumerate(top_countries, 1):
            code = country_code_by_name.get(country)
            ck = CountryKey(country=country, code=code)
            rng = stats.date_range_str()
            print(f"{i}. {ck.label()} — {stats.day_count()} days {rng}".rstrip())

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("🏙 Top destination cities")
    if not top_cities:
        print("No city data found.")
    else:
        for i, (pk, stats) in enumerate(top_cities, 1):
            home = " (home base)" if home_base == pk else ""
            rng = stats.date_range_str()
            print(f"{i}. {pk.city_label()} — {stats.day_count()} days {rng}{home}".rstrip())

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("📆 Month you traveled the most")
    if best_month:
        mstart, mend = month_day_ranges[best_month]
        rng = f"({_fmt_range(mstart, mend)})" if mstart and mend else ""
        print(f"• {_month_label(best_month)} — {len(month_cities[best_month])} cities, {len(month_countries[best_month])} countries")
        if rng:
            print(f"  {rng}")
    else:
        print("• (No month data found)")

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("🧳 Most traveled weeks")
    if not top_weeks:
        print("• (No multi-stop weeks found)")
    else:
        for wk in top_weeks:
            start = week_first_day.get(wk)
            end = week_last_day.get(wk)
            rng = f"{_fmt_range(start, end)}" if start and end else _iso_week_label(wk)
            route = " → ".join([pk.city for pk in week_places_order[wk]])
            print(f"• {rng}")
            print(f"  {route}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("✨ New places you discovered")
    if not new_places:
        print("• (No new places found)")
    else:
        for pk in new_places:
            rng = city_stats[pk].date_range_str()
            # Print “City, Country (Aug 10–14)”
            print(f"• {pk.city}, {pk.country} {rng}".rstrip())

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("🔁 Places you kept coming back to")
    if home_base:
        # Always list home base first
        hb_rng = city_stats[home_base].date_range_str()
        print(f"• {home_base.city}, {home_base.country} {hb_rng}".rstrip())
    if repeat_places:
        for pk in repeat_places:
            rng = city_stats[pk].date_range_str()
            months = sorted(city_months[pk])
            # show a compact month list like "(Mar, Jul, Aug)" without being too long
            month_names = []
            for mk in months[:6]:
                month_names.append(_month_label(mk)[:3])
            suffix = f"({', '.join(month_names)}" + (", …)" if len(months) > 6 else ")")
            print(f"• {pk.city}, {pk.country} {suffix}".rstrip())
    if not home_base and not repeat_places:
        print("• (No repeat places found)")

    print("\n━━━━━━━━━━━━━━━━━━━━━━")
    print("📸 Based on your photos:")
    print(f"• Photos scanned: {photos_scanned:,}")
    print(f"• Photos with location (lat/lon present): {photos_with_location:,}\n")
    print("Tip: Use the date ranges above to jump straight to those photos in Apple Photos.")
    print("Happy New Year 🎉")


if __name__ == "__main__":
    main()
