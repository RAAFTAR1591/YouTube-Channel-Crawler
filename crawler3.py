# Streamlit YouTube Channel Finder + Email Enricher
# ------------------------------------------------
# Features
# - Search channels by keyword within a recent window (days)
# - Optional geo search (lat,lng + radius) and/or country filter (ISO-2)
# - Filter by subscribers and uploads/week
# - Optional email enrichment from channel + recent video descriptions and external sites
# - Exports results to CSV
# - Clickable "Channel" link column for quick navigation
# ------------------------------------------------

import os
import re
import time
import json
import math
import urllib.parse as urlparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import requests
import pandas as pd
import streamlit as st
from urllib.robotparser import RobotFileParser

# ------------------------
# Utilities
# ------------------------
EMAIL_REGEX = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_REGEX = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>\"'()]+)""")


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="YouTube Channel Finder", layout="wide")
st.title("🔎 YouTube Channel Finder + Email Enricher")

with st.sidebar:
    st.header("API & Behavior")
    api_key = st.text_input(
        "YouTube Data API v3 Key",
        value=os.getenv("YOUTUBE_API_KEY", ""),
        type="password",
        help="Will not be saved. You can also set env var YOUTUBE_API_KEY."
    )
    base_url = "https://www.googleapis.com/youtube/v3"

    st.divider()
    st.caption("Search Settings")
    query = st.text_input("Keyword (query)", value="dentist")
    days = st.number_input("Lookback window (days)", min_value=1, max_value=365, value=30)
    min_subs = st.number_input("Min subscribers", min_value=0, value=100)
    max_subs = st.number_input("Max subscribers", min_value=0, value=500_000)

    st.caption("Uploads/Week Filter")
    min_uploads = st.number_input("Min uploads/week", min_value=0.0, step=0.1, value=0.0)
    max_uploads = st.number_input("Max uploads/week", min_value=0.0, step=0.5, value=50.0)

    st.caption("Geo Filter (optional, for geo-tagged videos)")
    lat = st.text_input("Latitude", value="")
    lng = st.text_input("Longitude", value="")
    radius = st.text_input("Radius (e.g., 50km)", value="")
    location = None
    location_radius = None
    if lat.strip() and lng.strip():
        location = f"{lat.strip()},{lng.strip()}"
    if radius.strip():
        location_radius = radius.strip()

    st.caption("Channel Country Filter (ISO-2 like IN, US)")
    country = st.text_input("Country code", value="") or None

    search_page_limit = st.slider("Search page limit", min_value=1, max_value=5, value=3,
                                  help="Each page returns up to 50 results")

    st.divider()
    enrich = st.checkbox("Enrich with emails from descriptions & external sites", value=False)
    scan_days = st.number_input("Scan recent videos (days)", min_value=7, max_value=365, value=60)
    per_site_delay = st.slider("Per-site polite delay (seconds)", min_value=0.0, max_value=3.0, value=1.0)
    pages_cap = st.slider("Max pages crawled per channel", min_value=1, max_value=20, value=10)
    respect_robots = st.checkbox("Respect robots.txt", value=True)

    st.divider()
    go = st.button("🚀 Search Channels")

# ------------------------
# Backend functions (cached where reasonable)
# ------------------------

def _published_after_iso(days_back: int) -> str:
    return (datetime.utcnow() - timedelta(days=days_back)).isoformat("T") + "Z"

@st.cache_data(show_spinner=False, ttl=600)
def search_videos(
    api_key: str,
    base_url: str,
    query: str,
    published_after: str,
    max_results: int = 50,
    location: Optional[str] = None,
    location_radius: Optional[str] = None,
    page_limit: int = 2,
) -> List[Dict]:
    url = f"{base_url}/search"
    items: List[Dict] = []
    token: Optional[str] = None
    pages = 0

    while pages < page_limit:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "date",
            "publishedAfter": published_after,
            "maxResults": min(50, max_results),
            "key": api_key,
        }
        if token:
            params["pageToken"] = token
        if location:
            params["location"] = location
        if location_radius:
            params["locationRadius"] = location_radius

        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("items", []))
        token = data.get("nextPageToken")
        pages += 1
        if not token:
            break
    return items

@st.cache_data(show_spinner=False, ttl=600)
def get_channel_details(api_key: str, base_url: str, channel_ids: List[str]) -> List[Dict]:
    url = f"{base_url}/channels"
    all_items: List[Dict] = []
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i:i+50]
        params = {
            "part": "snippet,statistics",
            "id": ",".join(chunk),
            "key": api_key,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        all_items.extend(data.get("items", []))
    return all_items


def safe_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def find_channels(
    api_key: str,
    base_url: str,
    query: str,
    min_subs: int,
    max_subs: int,
    days: int,
    location: Optional[str],
    location_radius: Optional[str],
    country: Optional[str],
    min_uploads_per_week: float,
    max_uploads_per_week: float,
    search_page_limit: int,
) -> List[Dict]:
    published_after = _published_after_iso(days)
    items = search_videos(
        api_key=api_key,
        base_url=base_url,
        query=query,
        published_after=published_after,
        max_results=50,
        location=location,
        location_radius=location_radius,
        page_limit=search_page_limit,
    )

    channel_uploads: Dict[str, int] = {}
    for it in items:
        sn = it.get("snippet", {})
        cid = sn.get("channelId")
        if not cid:
            continue
        channel_uploads[cid] = channel_uploads.get(cid, 0) + 1

    if not channel_uploads:
        return []

    channel_ids = list(channel_uploads.keys())
    details = get_channel_details(api_key, base_url, channel_ids)

    weeks = max(1e-9, days / 7.0)
    out: List[Dict] = []
    for ch in details:
        sn = ch.get("snippet", {})
        stt = ch.get("statistics", {})
        cid = ch.get("id")
        subs = safe_int(stt.get("subscriberCount"))
        if subs is None:
            continue
        uploads = channel_uploads.get(cid, 0)
        rate = uploads / weeks
        ch_country = sn.get("country")

        if not (min_subs <= subs <= max_subs):
            continue
        if not (min_uploads_per_week <= rate <= max_uploads_per_week):
            continue
        if country and ch_country and ch_country.upper() != country.upper():
            continue

        out.append({
            "title": sn.get("title"),
            "channelId": cid,
            "subscribers": subs,
            "upload_rate_per_week": round(rate, 2),
            "description": sn.get("description", ""),
            "country": ch_country,
            "uploads_count_in_window": uploads,
            "window_days": days,
        })

    out.sort(key=lambda x: (x["upload_rate_per_week"], x["subscribers"]), reverse=True)
    return out

# ------------------------
# Email enrichment helpers
# ------------------------

def extract_emails(text: str) -> Set[str]:
    if not text:
        return set()
    return set(EMAIL_REGEX.findall(text))


def extract_urls(text: str) -> Set[str]:
    if not text:
        return set()
    urls = set(URL_REGEX.findall(text))
    normed = set()
    for u in urls:
        if u.startswith("www."):
            u = "https://" + u
        host = (urlparse.urlparse(u).hostname or "").lower()
        if "youtube.com" in host or "youtu.be" in host:
            continue
        normed.add(u.rstrip(").,;"))
    return normed


def robots_allows(url: str, user_agent: str = "ChannelEmailFinder/1.0") -> bool:
    try:
        parts = urlparse.urlparse(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        path = parts.path or "/"
        return rp.can_fetch(user_agent, path)
    except Exception:
        # If robots is unreachable, be conservative if respecting robots
        return False


def fetch(url: str, timeout: int = 10, respect_robots: bool = True) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (ChannelEmailFinder/1.0)"}
    if respect_robots and not robots_allows(url):
        return None
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "text" in r.headers.get("Content-Type", ""):
            return r.text
    except Exception:
        return None
    return None


def domain_root(u: str) -> str:
    p = urlparse.urlparse(u)
    return f"{p.scheme}://{p.netloc}"


def find_on_site_links(html: str, base: str) -> Set[str]:
    links = set(re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE))
    out = set()
    for href in links:
        if href.startswith("#") or href.startswith("mailto:"):
            continue
        full = urlparse.urljoin(base, href)
        host = (urlparse.urlparse(full).hostname or "").lower()
        base_host = (urlparse.urlparse(base).hostname or "").lower()
        if host != base_host:
            continue
        if any(k in full.lower() for k in ["contact", "about", "support", "privacy", "impressum"]):
            out.add(full)
    return out


@st.cache_data(show_spinner=False, ttl=600)
def get_channel_snippet(api_key: str, base_url: str, channel_id: str) -> Optional[Dict]:
    url = f"{base_url}/channels"
    params = {"part": "snippet", "id": channel_id, "key": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = data.get("items", [])
    return items[0] if items else None


@st.cache_data(show_spinner=False, ttl=600)
def search_channel_recent_videos(api_key: str, base_url: str, channel_id: str, days: int = 30, max_pages: int = 2) -> List[Dict]:
    published_after = _published_after_iso(days)
    url = f"{base_url}/search"
    items: List[Dict] = []
    token = None
    pages = 0
    while pages < max_pages:
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "publishedAfter": published_after,
            "maxResults": 50,
            "key": api_key,
        }
        if token:
            params["pageToken"] = token
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("items", []))
        token = data.get("nextPageToken")
        pages += 1
        if not token:
            break
    return items


def enrich_channels_with_emails(
    api_key: str,
    base_url: str,
    channels: List[Dict],
    scan_days: int,
    per_site_delay_sec: float,
    pages_cap: int,
    respect_robots_flag: bool,
) -> List[Dict]:
    results = []
    for idx, ch in enumerate(channels, start=1):
        cid = ch.get("channelId")
        title = ch.get("title")
        st.write(f"🔗 Enriching [{idx}/{len(channels)}]: **{title}** ({cid})")
        found_emails: Set[str] = set()
        discovered_urls: Set[str] = set()

        # Channel description
        sn = get_channel_snippet(api_key, base_url, cid)
        ch_desc = sn.get("snippet", {}).get("description", "") if sn else ""
        found_emails |= extract_emails(ch_desc)
        discovered_urls |= extract_urls(ch_desc)

        # Recent videos
        vids = search_channel_recent_videos(api_key, base_url, cid, days=scan_days, max_pages=2)
        for v in vids:
            vdesc = v.get("snippet", {}).get("description", "") or ""
            found_emails |= extract_emails(vdesc)
            discovered_urls |= extract_urls(vdesc)

        # Crawl external sites
        site_emails: Set[str] = set()
        visited = set()
        queue: List[str] = []

        # Seed: roots and common endpoints
        for u in list(discovered_urls)[:5]:
            root = domain_root(u)
            for sfx in ["/", "/contact", "/about"]:
                q = urlparse.urljoin(root, sfx)
                if q not in queue:
                    queue.append(q)

        while queue and len(visited) < pages_cap:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            html = fetch(url, timeout=10, respect_robots=respect_robots_flag)
            time.sleep(per_site_delay_sec)
            if not html:
                continue
            site_emails |= extract_emails(html)
            # discover a few relevant on-site links
            for nxt in list(find_on_site_links(html, base=domain_root(url)))[:3]:
                if nxt not in visited:
                    queue.append(nxt)

        combined = sorted((found_emails | site_emails))
        results.append({
            **ch,
            "emails": ", ".join(combined) if combined else "",
            "email_sources": json.dumps({
                "channel_description": bool(extract_emails(ch_desc)),
                "video_descriptions_checked": len(vids),
                "external_urls_considered": list(discovered_urls)[:10],
                "pages_crawled": len(visited),
            }),
        })
    return results

# ------------------------
# Run search
# ------------------------
if go:
    if not api_key:
        st.error("Please provide your YouTube Data API key in the sidebar.")
        st.stop()

    with st.spinner("Searching videos and collecting channels..."):
        try:
            channels = find_channels(
                api_key=api_key,
                base_url=base_url,
                query=query,
                min_subs=int(min_subs),
                max_subs=int(max_subs),
                days=int(days),
                location=location,
                location_radius=location_radius,
                country=country,
                min_uploads_per_week=float(min_uploads),
                max_uploads_per_week=float(max_uploads),
                search_page_limit=int(search_page_limit),
            )
        except requests.HTTPError as e:
            st.exception(e)
            st.stop()

    if not channels:
        st.warning("No channels matched your filters. Try relaxing filters or increasing pages.")
        st.stop()

    df = pd.DataFrame(channels)

    # >>> Add a clickable Channel link column <<<
    df["channel_url"] = df["channelId"].apply(lambda cid: f"https://www.youtube.com/channel/{cid}")

    st.success(f"Found {len(df)} channels")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "channel_url": st.column_config.LinkColumn(
                "Channel",
                help="Open channel on YouTube",
                display_text="🔗 Open"
            ),
            "channelId": st.column_config.TextColumn("Channel ID"),
            "title": st.column_config.TextColumn("Title"),
            "subscribers": st.column_config.NumberColumn("Subscribers", format="%d"),
            "upload_rate_per_week": st.column_config.NumberColumn("Uploads/week"),
            "uploads_count_in_window": st.column_config.NumberColumn("Uploads in window"),
            "window_days": st.column_config.NumberColumn("Window (days)"),
            "country": st.column_config.TextColumn("Country"),
            "description": st.column_config.TextColumn("Description"),
        }
    )

    csv = df.drop(columns=["channel_url"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download channels CSV", data=csv, file_name="channels.csv", mime="text/csv")

    # Optional enrichment
    if enrich:
        st.subheader("Email Enrichment")
        with st.spinner("Scanning descriptions and external sites for emails..."):
            enriched = enrich_channels_with_emails(
                api_key=api_key,
                base_url=base_url,
                channels=channels,
                scan_days=int(scan_days),
                per_site_delay_sec=float(per_site_delay),
                pages_cap=int(pages_cap),
                respect_robots_flag=bool(respect_robots),
            )
        df_en = pd.DataFrame(enriched)

        # Ensure the link column exists on enriched DataFrame as well
        if "channel_url" not in df_en.columns:
            df_en["channel_url"] = df_en["channelId"].apply(lambda cid: f"https://www.youtube.com/channel/{cid}")

        st.success("Enrichment complete")
        st.dataframe(
            df_en,
            use_container_width=True,
            hide_index=True,
            column_config={
                "channel_url": st.column_config.LinkColumn(
                    "Channel",
                    help="Open channel on YouTube",
                    display_text="🔗 Open"
                ),
                "channelId": st.column_config.TextColumn("Channel ID"),
                "title": st.column_config.TextColumn("Title"),
                "subscribers": st.column_config.NumberColumn("Subscribers", format="%d"),
                "upload_rate_per_week": st.column_config.NumberColumn("Uploads/week"),
                "uploads_count_in_window": st.column_config.NumberColumn("Uploads in window"),
                "window_days": st.column_config.NumberColumn("Window (days)"),
                "country": st.column_config.TextColumn("Country"),
                "description": st.column_config.TextColumn("Description"),
                "emails": st.column_config.TextColumn("Emails"),
                "email_sources": st.column_config.TextColumn("Email sources (JSON)"),
            }
        )

        csv_en = df_en.drop(columns=["channel_url"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download enriched CSV", data=csv_en, file_name="channels_enriched.csv", mime="text/csv")

# Footer note
st.caption("Use responsibly. Do not bypass YouTube ToS. Respect robots.txt when crawling external sites.")
