# app.py — HCMC Bus Reachability (Interactive Demo)
# Fixes:
# - Basemap selection stored in st.session_state (won't reset when sliders change)
# - Default basemap = Light (CartoDB Positron)
# - All stops default ON and will NOT be auto-turned-off after selecting origin
# - Point size controls persist; map style persists across reruns

import os
import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import Fullscreen, LocateControl
from streamlit_folium import st_folium


# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = os.path.dirname(__file__)

STOPS_CSV = os.path.join(DATA_DIR, "stops.csv")
REACH_30 = os.path.join(DATA_DIR, "reachable_30.csv")
REACH_60 = os.path.join(DATA_DIR, "reachable_60.csv")
ROUTE_GEOJSON = os.path.join(DATA_DIR, "route_shape_lines.geojson")  # optional

DEFAULT_CENTER = (10.7769, 106.7009)  # HCMC approx
CLICK_NEAREST_THRESHOLD_M = 150  # within ~150m

# time bins + colors (green -> red)
TIME_BINS = [0, 10, 20, 30, 45, 60, 999]
TIME_LABELS = ["0–10", "10–20", "20–30", "30–45", "45–60", "60+"]
TIME_COLORS = ["#2ecc71", "#a3e635", "#facc15", "#fb923c", "#f97316", "#ef4444"]


# -----------------------------
# UTILS
# -----------------------------
def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def load_stops_and_routes():
    df = pd.read_csv(STOPS_CSV)
    cols = {c.lower(): c for c in df.columns}

    stopid_col = cols.get("stopid", None)
    lat_col = cols.get("lat", None)
    lng_col = cols.get("lng", None)
    if stopid_col is None or lat_col is None or lng_col is None:
        raise ValueError("stops.csv must contain StopId, Lat, Lng columns")

    name_col = None
    for cand in ["stopname", "name", "stop_name", "stop_name_en", "stop_name_vi"]:
        if cand in cols:
            name_col = cols[cand]
            break

    route_id_col = cols.get("routeid", None)
    route_var_col = cols.get("routevarid", None)

    route_name_col = None
    for cand in ["routename", "route_name", "routeno", "route_no", "shortname", "route_short_name"]:
        if cand in cols:
            route_name_col = cols[cand]
            break

    def make_route_label(r):
        if route_name_col is not None and pd.notna(r.get(route_name_col)):
            return str(r.get(route_name_col)).strip()
        if route_id_col is not None and route_var_col is not None:
            rid = r.get(route_id_col)
            rvid = r.get(route_var_col)
            if pd.notna(rid) and pd.notna(rvid):
                return f"{int(rid)}-{int(rvid)}"
        if route_id_col is not None:
            rid = r.get(route_id_col)
            if pd.notna(rid):
                return str(int(rid))
        return None

    df["_route_label"] = df.apply(make_route_label, axis=1)

    agg = []
    for sid, g in df.groupby(stopid_col):
        sid = int(sid)
        lat = float(g.iloc[0][lat_col])
        lng = float(g.iloc[0][lng_col])
        if name_col is not None and pd.notna(g.iloc[0][name_col]):
            sname = str(g.iloc[0][name_col]).strip()
        else:
            sname = f"Stop {sid}"

        route_labels = [x for x in g["_route_label"].dropna().astype(str).tolist() if x.strip() != ""]
        route_labels_unique = list(dict.fromkeys(route_labels))
        routes_str = ", ".join(route_labels_unique) if route_labels_unique else "N/A"

        agg.append(
            {"StopId": sid, "StopName": sname, "lat": lat, "lng": lng, "routes_str": routes_str}
        )

    stop_lookup = pd.DataFrame(agg).set_index("StopId").sort_index()
    return stop_lookup


@st.cache_data(show_spinner=False)
def load_reach_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    origin_col = None
    for cand in ["origin_stopid", "origin", "stopid", "origin_stop_id"]:
        if cand in cols:
            origin_col = cols[cand]
            break
    if origin_col is None:
        raise ValueError(f"{os.path.basename(path)} missing origin_stopid-like column")

    reach_col = None
    for cand in ["reachable_stopids", "reachable", "reachable_ids"]:
        if cand in cols:
            reach_col = cols[cand]
            break
    if reach_col is None:
        raise ValueError(f"{os.path.basename(path)} missing reachable_stopids-like column")

    minutes_col = None
    for cand in ["reachable_minutes", "minutes", "travel_minutes", "tt_minutes"]:
        if cand in cols:
            minutes_col = cols[cand]
            break

    out = df[[origin_col, reach_col] + ([minutes_col] if minutes_col else [])].copy()
    out = out.rename(columns={origin_col: "origin_stopid", reach_col: "reachable_stopids"})
    if minutes_col:
        out = out.rename(columns={minutes_col: "reachable_minutes"})
    return out


def parse_semicolon_int_list(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(int(float(p)))
        except:
            pass
    return out


def parse_semicolon_float_list(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            out.append(np.nan)
    return out


def time_to_color(mins):
    for i in range(len(TIME_BINS) - 1):
        if TIME_BINS[i] <= mins < TIME_BINS[i + 1]:
            return TIME_COLORS[i], TIME_LABELS[i]
    return TIME_COLORS[-1], TIME_LABELS[-1]


def build_legend():
    items = []
    for col, lab in zip(TIME_COLORS, TIME_LABELS):
        items.append(
            f"<div style='margin:2px 0; color:#111;'>"
            f"<span style='display:inline-block;width:12px;height:12px;background:{col};margin-right:6px;border-radius:2px;'></span>"
            f"{lab} min</div>"
        )

    # moved up + force black text so it never becomes white
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 90px; left: 30px;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        color: #111 !important;
        padding: 10px 12px;
        border: 1px solid #bbb;
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.25;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    ">
      <div style="font-weight:700; margin-bottom:6px; color:#111 !important;">
        Travel time (min)
      </div>
      <div style="color:#111 !important;">
        {''.join(items)}
      </div>
    </div>
    """
    return legend_html


def nearest_stop_id(stop_lookup, click_lat, click_lng):
    lats = stop_lookup["lat"].values
    lngs = stop_lookup["lng"].values
    d = haversine_m(click_lng, click_lat, lngs, lats)
    idx = int(np.argmin(d))
    sid = int(stop_lookup.index.values[idx])
    return sid, float(d[idx])


def build_reachable_table(reach_df, origin_id, stop_lookup, limit_min):
    row = reach_df[reach_df["origin_stopid"] == origin_id]
    if row.empty:
        return pd.DataFrame(columns=["StopId", "minutes"])

    reachable_ids = parse_semicolon_int_list(row.iloc[0]["reachable_stopids"])

    # If minutes exist, use them
    if "reachable_minutes" in reach_df.columns and pd.notna(row.iloc[0].get("reachable_minutes", np.nan)):
        mins_list = parse_semicolon_float_list(row.iloc[0]["reachable_minutes"])
        n = min(len(reachable_ids), len(mins_list))
        out = pd.DataFrame({"StopId": reachable_ids[:n], "minutes": mins_list[:n]}).dropna()
        out = out[out["minutes"] <= float(limit_min)].copy()
        return out.sort_values("minutes")

    # Otherwise approximate via distance
    o = stop_lookup.loc[origin_id]
    SPEED_M_PER_MIN = 333.0  # ~20km/h
    rows = []
    for sid in reachable_ids:
        if sid not in stop_lookup.index:
            continue
        d = haversine_m(o["lng"], o["lat"], stop_lookup.loc[sid, "lng"], stop_lookup.loc[sid, "lat"])
        rows.append((sid, d / SPEED_M_PER_MIN))
    out = pd.DataFrame(rows, columns=["StopId", "minutes"])
    out = out[out["minutes"] <= float(limit_min)].copy()
    return out.sort_values("minutes")


def add_basemaps(m, basemap_choice):
    """
    Keep basemap stable across reruns.
    We create Map(tiles=None) and add layers with show=(choice==...).
    """
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light (Carto)",
        control=True,
        show=(basemap_choice == "Light (Carto)"),
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        show=(basemap_choice == "OpenStreetMap"),
    ).add_to(m)

    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark (Carto)",
        control=True,
        show=(basemap_choice == "Dark (Carto)"),
    ).add_to(m)


def build_map(
    stop_lookup,
    origin_id,
    reachable_df,
    basemap_choice,
    show_all_stops,
    show_reachable,
    show_network,
    all_stop_radius,
    reach_stop_radius
):
    # Center
    if origin_id in stop_lookup.index:
        center = (float(stop_lookup.loc[origin_id, "lat"]), float(stop_lookup.loc[origin_id, "lng"]))
    else:
        center = DEFAULT_CENTER

    # IMPORTANT: tiles=None so we fully control default basemap via TileLayer(show=True)
    m = folium.Map(location=center, zoom_start=12, tiles=None, control_scale=True)

    add_basemaps(m, basemap_choice)

    Fullscreen().add_to(m)
    LocateControl(auto_start=False).add_to(m)

    # Optional network
    if show_network and os.path.exists(ROUTE_GEOJSON):
        try:
            folium.GeoJson(
                ROUTE_GEOJSON,
                name="Bus network (route lines)",
                style_function=lambda x: {"color": "#2563eb", "weight": 1.5, "opacity": 0.35},
            ).add_to(m)
        except Exception:
            pass

    fg_all = folium.FeatureGroup(name="All stops", show=show_all_stops)
    fg_reach = folium.FeatureGroup(name="Reachable stops (time-colored)", show=show_reachable)

    # All stops
    if show_all_stops:
        for sid, row in stop_lookup.iterrows():
            popup = folium.Popup(
                f"<b>{row['StopName']}</b><br>"
                f"StopId: {sid}<br>"
                f"Routes: {row['routes_str']}",
                max_width=380
            )
            folium.CircleMarker(
                location=(row["lat"], row["lng"]),
                radius=float(all_stop_radius),
                color="#111827",
                fill=True,
                fill_opacity=0.70,
                popup=popup,
            ).add_to(fg_all)

    # Origin marker
    if origin_id in stop_lookup.index:
        o = stop_lookup.loc[origin_id]
        folium.Marker(
            location=(o["lat"], o["lng"]),
            tooltip=f"Origin: {o['StopName']} (StopId={origin_id})",
            popup=folium.Popup(
                f"<b>Origin: {o['StopName']}</b><br>"
                f"StopId: {origin_id}<br>"
                f"Routes: {o['routes_str']}",
                max_width=380
            ),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    # Reachable stops
    if show_reachable and reachable_df is not None and len(reachable_df) > 0:
        for _, r in reachable_df.iterrows():
            sid = int(r["StopId"])
            mins = float(r["minutes"])
            if sid not in stop_lookup.index:
                continue
            row = stop_lookup.loc[sid]
            col, lab = time_to_color(mins)
            popup = folium.Popup(
                f"<b>{row['StopName']}</b><br>"
                f"StopId: {sid}<br>"
                f"Travel time: {mins:.1f} min ({lab})<br>"
                f"Routes: {row['routes_str']}",
                max_width=380
            )
            folium.CircleMarker(
                location=(row["lat"], row["lng"]),
                radius=float(reach_stop_radius),
                color=col,
                fill=True,
                fill_color=col,
                fill_opacity=0.92,
                popup=popup,
                tooltip=f"{row['StopName']} ({mins:.1f} min)"
            ).add_to(fg_reach)

    fg_all.add_to(m)
    fg_reach.add_to(m)

    # legend
    m.get_root().html.add_child(folium.Element(build_legend()))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# -----------------------------
# APP
# -----------------------------
st.set_page_config(page_title="HCMC Bus Reachability Demo", layout="wide")

st.title("HCMC Bus Reachability (Interactive Demo)")
st.caption(
    "Click near a stop to select the nearest stop (within ~150m). "
    "Use search + dropdown to choose origin. Reachable stops are time-colored."
)

if not os.path.exists(STOPS_CSV):
    st.error(f"Missing file: {STOPS_CSV}")
    st.stop()

stop_lookup = load_stops_and_routes()

# session defaults
if "origin_id" not in st.session_state:
    st.session_state.origin_id = int(stop_lookup.index.values[0])

if "show_all_stops" not in st.session_state:
    st.session_state.show_all_stops = True

if "basemap" not in st.session_state:
    st.session_state.basemap = "Light (Carto)"  # ✅ always default light, and persist later


# Sidebar
st.sidebar.header("Controls")

# ✅ default 30 min
limit_min = st.sidebar.radio("Time limit (minutes)", [30, 60], index=0)

# ✅ basemap choice (persist)
basemap_choice = st.sidebar.radio(
    "Basemap",
    ["Light (Carto)", "OpenStreetMap", "Dark (Carto)"],
    index=["Light (Carto)", "OpenStreetMap", "Dark (Carto)"].index(st.session_state.basemap),
)
st.session_state.basemap = basemap_choice

show_network = st.sidebar.checkbox("Show bus network (route lines)", value=False)
show_reachable = st.sidebar.checkbox("Reachable stops (time-colored)", value=True)

# point size sliders
st.sidebar.subheader("Point sizes")
all_stop_radius = st.sidebar.slider("All stops point size", 1.0, 8.0, 3.0, 0.5)
reach_stop_radius = st.sidebar.slider("Reachable stops point size", 2.0, 12.0, 5.0, 0.5)

# ✅ All stops default ON and persist; no auto-off anymore
show_all_stops = st.sidebar.checkbox("Show all stops (points)", value=st.session_state.show_all_stops)
st.session_state.show_all_stops = show_all_stops

st.sidebar.divider()
st.sidebar.subheader("Search origin stop")
kw = st.sidebar.text_input("Type stop name keywords", value="", placeholder="e.g., Ben Thanh, Nguyen...")

labels, ids = [], []
for sid, r in stop_lookup.iterrows():
    label = f"{r['StopName']} (StopId={sid})"
    if kw.strip() and kw.lower() not in r["StopName"].lower():
        continue
    labels.append(label)
    ids.append(int(sid))

if not ids:
    st.sidebar.warning("No stops match your search.")
    ids = [int(stop_lookup.index.values[0])]
    r = stop_lookup.loc[ids[0]]
    labels = [f"{r['StopName']} (StopId={ids[0]})"]

current_origin = st.session_state.origin_id
default_index = ids.index(current_origin) if current_origin in ids else 0
selected_label = st.sidebar.selectbox("Choose origin stop", labels, index=default_index)
selected_id = int(ids[labels.index(selected_label)])

# Update origin (no auto-off all-stops)
if selected_id != st.session_state.origin_id:
    st.session_state.origin_id = selected_id

origin_id = st.session_state.origin_id

# load reach data
reach_df = load_reach_csv(REACH_60 if limit_min == 60 else REACH_30)
reachable_table = build_reachable_table(reach_df, origin_id, stop_lookup, limit_min)

# summary
o = stop_lookup.loc[origin_id]
c1, c2, c3 = st.columns([1.2, 1, 2])
with c1:
    st.metric("Origin stop", o["StopName"])
    st.caption(f"StopId: {origin_id}")
with c2:
    st.metric("Reachable stops", int(len(reachable_table)))
    st.caption(f"Within {limit_min} min")
with c3:
    routes_preview = o["routes_str"]
    if len(routes_preview) > 200:
        routes_preview = routes_preview[:200] + " ..."
    st.markdown("**Routes at origin**")
    st.write(routes_preview)

st.subheader("Map")
st.caption("Tip: click near a stop to select it. Map style will stay the same when you change controls.")

m = build_map(
    stop_lookup=stop_lookup,
    origin_id=origin_id,
    reachable_df=reachable_table,
    basemap_choice=st.session_state.basemap,
    show_all_stops=st.session_state.show_all_stops,
    show_reachable=show_reachable,
    show_network=show_network,
    all_stop_radius=all_stop_radius,
    reach_stop_radius=reach_stop_radius
)

out = st_folium(m, height=650, width=None, returned_objects=["last_clicked"])

# click -> nearest stop
if out and out.get("last_clicked"):
    clat = out["last_clicked"]["lat"]
    clng = out["last_clicked"]["lng"]
    sid, dist_m = nearest_stop_id(stop_lookup, clat, clng)
    if dist_m <= CLICK_NEAREST_THRESHOLD_M and sid != st.session_state.origin_id:
        st.session_state.origin_id = int(sid)
        st.rerun()

with st.expander("Show reachable stops table"):
    if len(reachable_table) == 0:
        st.info("No reachable stops found for this origin/time limit (or the origin isn't present in the reach file).")
    else:
        tmp = reachable_table.copy()
        tmp["StopName"] = tmp["StopId"].map(
            lambda x: stop_lookup.loc[int(x), "StopName"] if int(x) in stop_lookup.index else ""
        )
        tmp["Routes"] = tmp["StopId"].map(
            lambda x: stop_lookup.loc[int(x), "routes_str"] if int(x) in stop_lookup.index else ""
        )
        st.dataframe(tmp[["StopId", "StopName", "minutes", "Routes"]].head(500), use_container_width=True)