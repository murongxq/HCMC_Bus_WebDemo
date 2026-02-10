# prep_reach_time.py
# ------------------------------------------------------------
# Compute min travel time (sec/min) from each origin stop to reachable stops
# using your bus_access_grid_v2 logic:
# - route speed from geometry length / TimeOfTrip (fallback speed)
# - dwell time at each stop
# - expected waiting time = Headway/2 per boarding
# - transfer between nearby stops: walking time only
# - no first/last-mile walk to the first stop
# - origins are UNIQUE StopId
#
# Outputs (in outputs_time/):
#   reach_time_30.csv (long)
#   reach_time_60.csv (long)
#   reach_time_30_agg.csv (wide-ish, one origin per row)
#   reach_time_60_agg.csv
# ------------------------------------------------------------

import os
import re
import math
import heapq
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import substring
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

STOPS_FILE = BASE_DIR / "stops.csv"
ROUTE_FILE = BASE_DIR / "route_shape_lines.geojson"

OUT_DIR = BASE_DIR / "outputs_time"
OUT_DIR.mkdir(exist_ok=True)

TIME_LIMITS_MIN = [30, 60]

# ---------- Behavior parameters ----------
DWELL_SEC = 20.0

TRANSFER_RADIUS_M = 60.0
WALK_SPEED_MPS = 1.2

HEADWAY_FALLBACK_MIN = 15.0
FALLBACK_SPEED_MPS = 5.0  # 18 km/h ideal-ish

MIN_SEG_LEN_M = 1.0

CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32648"  # HCMC


def done(msg):  print(f"✅ DONE: {msg}")
def info(msg):  print(f"ℹ️  INFO: {msg}")
def check(msg): print(f"🔎 CHECK: {msg}")


def parse_minutes(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    nums = re.findall(r"[\d.]+", s)
    if not nums:
        return None
    nums = [float(x) for x in nums]
    if len(nums) == 1:
        return nums[0]
    return sum(nums) / len(nums)


def safe_speed_mps(route_row):
    length_m = route_row.get("geom_len_m", None)
    if length_m is None or length_m <= 0:
        return FALLBACK_SPEED_MPS

    cand_cols = ["TimeOfTrip", "RunningTime", "TimeTrip", "TimeofTrip", "TimeOfTripMin"]
    minutes = None
    for c in cand_cols:
        if c in route_row and route_row[c] is not None:
            minutes = parse_minutes(route_row[c])
            if minutes is not None and minutes > 0:
                break

    if minutes is None or minutes <= 0:
        return FALLBACK_SPEED_MPS

    return max(length_m / (minutes * 60.0), 0.1)


def expected_wait_sec(route_row):
    cand_cols = ["Headway", "HeadWay", "headway", "HeadwayMin"]
    hw_min = None
    for c in cand_cols:
        if c in route_row and route_row[c] is not None:
            hw_min = parse_minutes(route_row[c])
            if hw_min is not None and hw_min > 0:
                break
    if hw_min is None or hw_min <= 0:
        hw_min = HEADWAY_FALLBACK_MIN
    return float(hw_min) * 60.0 / 2.0


def build_stop_points(stops_df):
    lat_col = "Lat" if "Lat" in stops_df.columns else ("lat" if "lat" in stops_df.columns else None)
    lng_col = "Lng" if "Lng" in stops_df.columns else ("lng" if "lng" in stops_df.columns else None)
    if lat_col is None or lng_col is None:
        raise ValueError("stops.csv missing Lat/Lng (or lat/lng) columns")

    gdf = gpd.GeoDataFrame(
        stops_df.copy(),
        geometry=gpd.points_from_xy(stops_df[lng_col], stops_df[lat_col]),
        crs=CRS_WGS84
    ).to_crs(CRS_UTM)

    # IMPORTANT: keep StopId, RouteId, RouteVarId
    for c in ["StopId", "RouteId", "RouteVarId"]:
        if c not in gdf.columns:
            raise ValueError(f"stops.csv missing column: {c}")
    gdf["StopId"] = gdf["StopId"].astype(int)
    gdf["RouteId"] = gdf["RouteId"].astype(int)
    gdf["RouteVarId"] = gdf["RouteVarId"].astype(int)
    return gdf


def build_route_lookup(routes_gdf):
    for c in ["RouteId", "RouteVarId"]:
        if c not in routes_gdf.columns:
            raise ValueError(f"route geojson missing column: {c}")

    route_lookup = {}
    dup = 0
    for _, r in routes_gdf.iterrows():
        key = (int(r["RouteId"]), int(r["RouteVarId"]))
        if key in route_lookup:
            dup += 1
            continue
        route_lookup[key] = r
    check(f"Route lookup keys={len(route_lookup)}, duplicated_skipped={dup}")
    return route_lookup


def project_stoprows_to_routes(stops_gdf, route_lookup):
    records = []
    failed = 0
    for _, s in stops_gdf.iterrows():
        key = (int(s["RouteId"]), int(s["RouteVarId"]))
        if key not in route_lookup:
            failed += 1
            continue
        line = route_lookup[key].geometry
        pt = s.geometry
        m = line.project(pt)
        proj_pt = line.interpolate(m)
        records.append({
            "StopId": int(s["StopId"]),
            "RouteId": int(s["RouteId"]),
            "RouteVarId": int(s["RouteVarId"]),
            "route_key": key,
            "measure_m": float(m),
            "proj_pt": proj_pt
        })
    proj_df = pd.DataFrame(records)
    check(f"Projected stop-rows={len(proj_df)}, failed_no_route={failed}")
    return proj_df


# ---- Graph nodes: Platform and Route nodes ----
def P(stop_id):
    return ("P", int(stop_id))

def R(stop_id, rid, rvid):
    return ("R", int(stop_id), int(rid), int(rvid))


def build_graph(proj_df, route_lookup):
    """
    Nodes:
      P(stop) platform node
      R(stop, route) on-vehicle state node

    Edges:
      P -> R : wait = headway/2 (board)
      R -> R(next) : travel + dwell (ride)
      R -> P : 0 (alight)
      P -> P(nearby) : walk time (transfer walking only)
    """
    adj = {}

    def add_edge(u, v, t, etype, geom=None):
        adj.setdefault(u, []).append((v, float(t), etype, geom))

    # Platform representative point (for transfer)
    stop_pts = proj_df.groupby("StopId").first().reset_index()[["StopId", "proj_pt"]].copy()
    stop_pts_gdf = gpd.GeoDataFrame(stop_pts, geometry="proj_pt", crs=CRS_UTM).rename(columns={"proj_pt": "geometry"})
    stop_pts_gdf = stop_pts_gdf.set_geometry("geometry")

    # P <-> R edges
    for _, row in proj_df[["StopId", "RouteId", "RouteVarId", "route_key", "measure_m"]].iterrows():
        sid = int(row["StopId"])
        rid = int(row["RouteId"])
        rvid = int(row["RouteVarId"])
        rr = route_lookup[(rid, rvid)]
        wsec = expected_wait_sec(rr)
        add_edge(P(sid), R(sid, rid, rvid), wsec, "board_wait", None)
        add_edge(R(sid, rid, rvid), P(sid), 0.0, "alight", None)

    # Ride edges along each route (directed by measure order)
    info("Building ride edges ...")
    ride_cnt = 0
    for (rid, rvid), grp in proj_df.groupby(["RouteId", "RouteVarId"]):
        rid = int(rid); rvid = int(rvid)
        key = (rid, rvid)
        line = route_lookup[key].geometry
        speed = safe_speed_mps(route_lookup[key])

        grp2 = grp.sort_values("measure_m").drop_duplicates(subset=["StopId"], keep="first")
        stop_ids = grp2["StopId"].tolist()
        measures = grp2["measure_m"].tolist()

        for i in range(len(stop_ids) - 1):
            a = int(stop_ids[i])
            b = int(stop_ids[i + 1])
            m1 = float(measures[i])
            m2 = float(measures[i + 1])
            if m2 <= m1:
                continue
            dist_m = m2 - m1
            if dist_m < MIN_SEG_LEN_M:
                continue
            travel_sec = dist_m / speed
            cost = travel_sec + DWELL_SEC
            try:
                seg = substring(line, m1, m2)
                if seg is None or seg.is_empty:
                    continue
                if not isinstance(seg, LineString):
                    continue
            except Exception:
                continue

            add_edge(R(a, rid, rvid), R(b, rid, rvid), cost, "ride", seg)
            ride_cnt += 1
    done(f"Ride edges built = {ride_cnt}")

    # Transfer walk edges between platform nodes
    info("Building transfer walk edges ...")
    sindex = stop_pts_gdf.sindex
    transfer_cnt = 0
    for idx, row in stop_pts_gdf.iterrows():
        sid_u = int(row["StopId"])
        pu = row.geometry
        if pu is None or pu.is_empty:
            continue
        buf = pu.buffer(TRANSFER_RADIUS_M)
        cand = list(sindex.intersection(buf.bounds))
        for j in cand:
            if j == idx:
                continue
            sid_v = int(stop_pts_gdf.iloc[j]["StopId"])
            pv = stop_pts_gdf.iloc[j].geometry
            if pv is None or pv.is_empty:
                continue
            d = pu.distance(pv)
            if d <= TRANSFER_RADIUS_M:
                walk_sec = d / WALK_SPEED_MPS
                add_edge(P(sid_u), P(sid_v), walk_sec, "transfer_walk", None)
                transfer_cnt += 1
    done(f"Transfer edges built = {transfer_cnt} (directed)")

    return adj, stop_pts_gdf


def dijkstra_time_limited(adjacency, origin_node, time_limit_sec):
    dist = {origin_node: 0.0}
    pq = [(0.0, origin_node)]
    while pq:
        t, u = heapq.heappop(pq)
        if t != dist.get(u, None):
            continue
        if t > time_limit_sec:
            continue
        for v, w, _, _ in adjacency.get(u, []):
            nt = t + w
            if nt > time_limit_sec:
                continue
            if v not in dist or nt < dist[v]:
                dist[v] = nt
                heapq.heappush(pq, (nt, v))
    return dist


def export_long_and_agg(origins, adjacency, time_min: int):
    tl_sec = float(time_min) * 60.0

    long_rows = []

    for sid in tqdm(origins, desc=f"Dijkstra origins @{time_min}min"):
        dist = dijkstra_time_limited(adjacency, P(sid), tl_sec)

        # reachable stops defined as platform nodes P(stop)
        for k, tsec in dist.items():
            if not (isinstance(k, tuple) and len(k) == 2 and k[0] == "P"):
                continue
            dest = int(k[1])
            if tsec <= tl_sec:
                long_rows.append((int(sid), int(dest), float(tsec), float(tsec) / 60.0))

    long_df = pd.DataFrame(long_rows, columns=["origin_stopid", "dest_stopid", "time_sec", "time_min"])
    out_long = OUT_DIR / f"reach_time_{time_min}.csv"
    long_df.to_csv(out_long, index=False)
    done(f"Saved long: {out_long}  (rows={len(long_df)})")

    # Aggregate for fast web lookup: each origin in one row with "dest;dest;..." and "t; t;..."
    agg = (
        long_df.sort_values(["origin_stopid", "time_sec"])
        .groupby("origin_stopid")
        .apply(lambda d: pd.Series({
            "dest_stopids": ";".join(map(str, d["dest_stopid"].tolist())),
            "time_secs": ";".join([f"{x:.1f}" for x in d["time_sec"].tolist()]),
            "reachable_count": int(len(d))
        }))
        .reset_index()
    )
    out_agg = OUT_DIR / f"reach_time_{time_min}_agg.csv"
    agg.to_csv(out_agg, index=False)
    done(f"Saved agg:  {out_agg}  (origins={len(agg)})")


def main():
    print("\n==================== STEP 1: LOAD ====================")
    routes = gpd.read_file(ROUTE_FILE).set_crs(CRS_WGS84, allow_override=True).to_crs(CRS_UTM)
    routes = routes[~routes.geometry.is_empty & routes.geometry.notna()].copy()
    routes["geom_len_m"] = routes.geometry.length
    done(f"Routes loaded: {len(routes)}")

    stops_df = pd.read_csv(STOPS_FILE)
    stops_gdf = build_stop_points(stops_df)
    done(f"Stops loaded (rows): {len(stops_gdf)}")

    print("\n==================== STEP 2: LOOKUP + PROJECT ====================")
    route_lookup = build_route_lookup(routes)
    proj_df = project_stoprows_to_routes(stops_gdf, route_lookup)
    if len(proj_df) == 0:
        raise RuntimeError("No stop-rows projected to routes. Check RouteId/RouteVarId matching.")
    done(f"Projected stop-rows: {len(proj_df)}")

    print("\n==================== STEP 3: BUILD GRAPH ====================")
    adjacency, stop_pts_gdf = build_graph(proj_df, route_lookup)
    origins = sorted(stop_pts_gdf["StopId"].unique().tolist())
    done(f"Origins (unique StopId) = {len(origins)}")

    print("\n==================== STEP 4: EXPORT TIME TABLES ====================")
    for tmin in TIME_LIMITS_MIN:
        export_long_and_agg(origins, adjacency, tmin)

    print("\n==================== ALL DONE ====================\n")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()