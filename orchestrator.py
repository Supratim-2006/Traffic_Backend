"""
orchestrator.py — Accident Analysis Orchestrator API
=====================================================
Accepts an image upload + location context, fans out to specialist
APIs in parallel, merges results, and returns a single dashboard-ready
JSON payload.

Service map
-----------
  Object Detection  POST /analyze   → https://supratimkukri-crowdflow.hf.space/analyze
  Road Closure      POST /predict   → https://supratimkukri-RoadClosure.hf.space/predict
  Disruption Class  POST /predict   → https://traffic-disruption.onrender.com/predict

Routing is independent — call the traffic routing API directly.

Run this orchestrator on port 8004:
  uvicorn orchestrator:app --host 0.0.0.0 --port 8004 --reload
"""

from __future__ import annotations

import asyncio
import random
import traceback
from datetime import datetime, timezone
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OBJECT_DETECTION_URL = "https://supratimkukri-crowdflow.hf.space/analyze"
ROAD_CLOSURE_URL     = "https://supratimkukri-RoadClosure.hf.space/predict"
DISRUPTION_URL       = "https://traffic-disruption.onrender.com/predict"

TIMEOUT = 60.0   # seconds per downstream call

# ─────────────────────────────────────────────────────────────────────────────
# EVENT CAUSE TAXONOMY  (derived from production event_cause value_counts)
# ─────────────────────────────────────────────────────────────────────────────
# Canonical reasons accepted from the client. Keys are normalized
# (lowercase, spaces/slashes -> underscore) so "Fog / Low Visibility"
# and "fog_low_visibility" both resolve correctly.
EVENT_CAUSE_PLANNED_MAP = {
    "vehicle_breakdown":   "unplanned",
    "others":              "unplanned",
    "pot_holes":           "unplanned",
    "construction":        "planned",
    "water_logging":       "unplanned",
    "accident":            "unplanned",
    "tree_fall":           "unplanned",
    "road_conditions":     "unplanned",
    "congestion":          "unplanned",
    "public_event":        "planned",
    "procession":          "planned",
    "vip_movement":        "planned",
    "protest":             "unplanned",
    "debris":              "unplanned",
    "test_demo":           "unplanned",
    "fog_low_visibility":  "unplanned",
}

# event_cause options surfaced to API consumers / docs (human-readable form).
VALID_EVENT_CAUSES = sorted(EVENT_CAUSE_PLANNED_MAP.keys())


def _normalize_cause(raw: str) -> str:
    return (
        raw.strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .strip("_")
    )


def _resolve_event_cause(raw: str) -> tuple[str, str]:
    """
    Normalize a free-text/road_block_reason into a canonical event_cause
    and resolve whether it's a planned or unplanned event.

    Returns (canonical_event_cause, planned_status) where planned_status
    is one of "planned" / "unplanned". Unrecognized reasons default to
    "others" / "unplanned" rather than failing the request, since the
    reason itself is still recorded verbatim downstream.
    """
    normalized = _normalize_cause(raw)
    if normalized in EVENT_CAUSE_PLANNED_MAP:
        return normalized, EVENT_CAUSE_PLANNED_MAP[normalized]
    return "others", "unplanned"


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrowdFlow AI — Accident Analysis Orchestrator",
    description=(
        "Uploads an accident image, calls all specialist APIs in parallel, "
        "and returns a single merged payload ready for the dashboard. "
        "Routing is independent — call the traffic routing API directly."
    ),
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# PAYLOAD BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_closure_payload(
    analysis: dict,
    latitude: float,
    longitude: float,
    zone: str,
    corridor: str,
    junction: Optional[str],
    event_cause: str,
    event_type: str,
) -> dict:
    """Build the exact payload expected by road_closure /predict."""
    emergency = analysis.get("emergency", {})
    em_level  = emergency.get("level", "LOW")
    veh_types = analysis.get("vehicle_types", {})

    priority_map = {"CRITICAL": "High", "HIGH": "High", "MEDIUM": "Low", "LOW": "Low"}

    dominant_veh = ""
    if veh_types:
        raw = max(veh_types, key=veh_types.get)
        dominant_veh = {
            "car": "Car", "truck": "Truck", "bus": "Bus",
            "motorbike": "Two-Wheeler", "van": "Van",
        }.get(raw, raw.title())

    return {
        "start_datetime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+0000"),
        "latitude":       latitude,
        "longitude":      longitude,
        "event_cause":    event_cause,
        "priority":       priority_map.get(em_level, "Low"),
        "zone":           zone,
        "corridor":       corridor,
        "event_type":     event_type,
        "veh_type":       dominant_veh,
        "junction":       junction,
    }


def _build_disruption_payload(analysis: dict, closure_payload: dict) -> dict:
    """Build the exact payload expected by traffic_disruption /predict."""
    emergency = analysis.get("emergency", {})
    reasons   = emergency.get("reasons", [])
    desc      = "; ".join(reasons[:2]) if reasons else "Traffic incident detected"
    comment   = (
        f"{analysis.get('vehicles', 0)} vehicle(s), "
        f"{analysis.get('people', 0)} person(s) at scene."
    )
    return {
        **closure_payload,
        "description":           desc,
        "comment":               comment,
        "requires_road_closure": True,
        "status":                "Open",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC CALLERS
# ─────────────────────────────────────────────────────────────────────────────

async def call_object_detection(
    client: httpx.AsyncClient,
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict:
    resp = await client.post(
        OBJECT_DETECTION_URL,
        files={"file": (filename, image_bytes, content_type)},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def call_road_closure(client: httpx.AsyncClient, payload: dict) -> dict:
    resp = await client.post(ROAD_CLOSURE_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


async def call_disruption(client: httpx.AsyncClient, payload: dict) -> dict:
    resp = await client.post(DISRUPTION_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_closure(result: dict) -> tuple[bool, float, str]:
    """Returns (road_closure_required, confidence, risk_level)."""
    if "_error" in result:
        return False, 0.0, "Unknown"
    pred = result.get("prediction", result)
    return (
        bool(pred.get("road_closure_required", False)),
        float(pred.get("confidence", 0.0)),
        str(pred.get("risk_level", "Unknown")),
    )


def _parse_disruption(result: dict) -> tuple[str, str, str]:
    """Returns (severity_label, confidence_str, eta_range)."""
    eta_map = {
        "<30 mins (Quick)":    "5 – 15 mins",
        "30–90 mins (Minor)":  "30 – 90 mins",
        "90–240 mins (Major)": "90 – 240 mins",
        ">240 mins (Severe)":  "240+ mins",
    }
    if "_error" in result:
        return "Unknown", "N/A", "Unknown"
    pred  = result.get("prediction", result)
    label = str(pred.get("label", "Unknown"))
    conf  = str(pred.get("confidence", "N/A"))
    return label, conf, eta_map.get(label, "Unknown")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _safe(label: str, result):
    """Wrap an exception in a tagged error dict; pass through normal results."""
    if isinstance(result, Exception):
        return {"_error": f"{label}: {result}"}
    return result


def _congestion_pct(score: float) -> int:
    return min(100, max(0, int(round(score * 100))))


def _generate_heatmap_points(
    lat: float, lon: float, congestion_score: float, n: int = 50
) -> list[list[float]]:
    spread = max(0.002, congestion_score * 0.025)
    return [
        [
            lat + random.gauss(0, spread),
            lon + random.gauss(0, spread),
            round(congestion_score * random.uniform(0.6, 1.0), 3),
        ]
        for _ in range(n)
    ]


def _generate_recommendations(
    em_level: str,
    road_closure_req: bool,
    severity_label: str,
    people: int,
    vehicles: int,
    congestion_score: float,
) -> list[str]:
    recs = []

    if em_level in ("CRITICAL", "HIGH"):
        recs.append("Deploy traffic police at scene immediately")
        recs.append("Dispatch emergency services (ambulance / fire)")

    if road_closure_req:
        recs.append("Implement road closure — use routing API for live diversion route")

    if people > 20:
        recs.append("Crowd control required — deploy marshals at perimeter")

    if vehicles > 10 or congestion_score > 0.7:
        recs.append("Increase public transport frequency on adjacent routes")

    if severity_label in ("90–240 mins (Major)", ">240 mins (Severe)"):
        recs.append("Inform public via variable message signs and radio")
        recs.append("Alert city traffic control centre")

    if not recs:
        recs.append("Monitor situation — no immediate action required")

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# POLICE PERSONNEL REQUIREMENT MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Brackets are deliberately simple/explainable rather than a black-box model,
# since dispatch decisions need to be auditable. Base headcount comes from
# congestion severity, then emergency level + planned/unplanned status, and
# scale (vehicles/people) act as escalators on top of the base bracket.

_CONGESTION_BASE_BRACKETS = [
    # (max_score_exclusive_upper_bound, bracket_label, base_personnel)
    (0.25, "Light (2-3 personnel)",        2),
    (0.50, "Moderate (3-5 personnel)",     3),
    (0.70, "Heavy (5-8 personnel)",        5),
    (0.85, "Severe (8-12 personnel)",      8),
    (1.01, "Critical (12-20 personnel)",  12),
]

_EMERGENCY_LEVEL_BONUS = {
    "LOW":      0,
    "MEDIUM":   1,
    "HIGH":     3,
    "CRITICAL": 6,
}

# Above this many required personnel, a single beat/patrol unit isn't enough —
# surface the nearest police station so dispatch can be raised to station level.
POLICE_STATION_LOOKUP_THRESHOLD = 6

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


async def _find_nearest_police_station(
    client: httpx.AsyncClient, lat: float, lon: float, radius_m: int = 8000
) -> Optional[dict]:
    """
    Query OpenStreetMap's Overpass API for the nearest amenity=police node
    within radius_m of (lat, lon). Returns None on failure or no results
    rather than raising — this is a best-effort enrichment, not a hard
    dependency of the response.
    """
    query = f"""
    [out:json][timeout:10];
    node["amenity"="police"](around:{radius_m},{lat},{lon});
    out body;
    """
    try:
        resp = await client.post(OVERPASS_URL, data={"data": query}, timeout=15.0)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        return None

    if not elements:
        return None

    def _dist(el: dict) -> float:
        # Equirectangular approximation — fine for short-range nearest-station ranking.
        dlat = el["lat"] - lat
        dlon = el["lon"] - lon
        return (dlat * dlat) + (dlon * dlon)

    nearest = min(elements, key=_dist)
    tags = nearest.get("tags", {})
    dlat = nearest["lat"] - lat
    dlon = nearest["lon"] - lon
    approx_km = round(((dlat * 111.0) ** 2 + (dlon * 111.0 * 0.96) ** 2) ** 0.5, 2)

    return {
        "name":           tags.get("name", "Unnamed Police Station"),
        "latitude":       nearest["lat"],
        "longitude":      nearest["lon"],
        "approx_distance_km": approx_km,
        "address":        ", ".join(
            v for v in [
                tags.get("addr:housenumber"),
                tags.get("addr:street"),
                tags.get("addr:suburb"),
                tags.get("addr:city"),
            ] if v
        ) or None,
        "phone":          tags.get("phone") or tags.get("contact:phone"),
        "source":         "OpenStreetMap (Overpass API)",
    }


def _police_personnel_required(
    congestion_score: float,
    em_level: str,
    event_type: str,
    vehicles: int,
    people: int,
) -> dict:
    """
    Estimate how many traffic police personnel are needed to clear the
    road block, and which dispatch bracket that falls into.
    """
    congestion_score = max(0.0, min(1.0, congestion_score))

    base = _CONGESTION_BASE_BRACKETS[-1]
    for upper, label, personnel in _CONGESTION_BASE_BRACKETS:
        if congestion_score < upper:
            base = (upper, label, personnel)
            break

    bracket_label, base_personnel = base[1], base[2]

    emergency_bonus = _EMERGENCY_LEVEL_BONUS.get(em_level.upper(), 0)

    # Unplanned events (accidents, breakdowns, debris, etc.) need faster,
    # heavier response than planned ones (processions, VIP movement,
    # scheduled construction) which are usually pre-staffed/coordinated.
    planned_bonus = 0 if event_type == "planned" else 1

    # Scale escalators: large vehicle pile-ups or crowds need more hands
    # on deck regardless of the raw congestion score.
    scale_bonus = 0
    if vehicles > 10:
        scale_bonus += 2
    elif vehicles > 5:
        scale_bonus += 1

    if people > 20:
        scale_bonus += 2
    elif people > 10:
        scale_bonus += 1

    total_personnel = base_personnel + emergency_bonus + planned_bonus + scale_bonus
    total_personnel = max(1, total_personnel)

    # Re-derive the dispatch bracket label from the final total so it
    # stays consistent even after bonuses push it past the base bracket.
    if total_personnel <= 3:
        final_bracket = "Light (2-3 personnel)"
    elif total_personnel <= 5:
        final_bracket = "Moderate (3-5 personnel)"
    elif total_personnel <= 8:
        final_bracket = "Heavy (5-8 personnel)"
    elif total_personnel <= 12:
        final_bracket = "Severe (8-12 personnel)"
    else:
        final_bracket = "Critical (12-20 personnel)"

    return {
        "personnel_required": total_personnel,
        "bracket":             final_bracket,
        "breakdown": {
            "congestion_base":  base_personnel,
            "emergency_bonus":  emergency_bonus,
            "event_type_bonus": planned_bonus,
            "scale_bonus":      scale_bonus,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "CrowdFlow AI Orchestrator v2.2"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "apis": {
            "object_detection": OBJECT_DETECTION_URL,
            "road_closure":     ROAD_CLOSURE_URL,
            "disruption":       DISRUPTION_URL,
        },
        "note": "Routing API is independent — call it directly for diversion routes.",
    }


@app.get("/event-causes", tags=["Reference"])
def event_causes():
    """List the valid road_block_reason values and whether each is planned/unplanned."""
    return {"event_causes": EVENT_CAUSE_PLANNED_MAP}


@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file:               UploadFile = File(..., description="Accident scene image"),
    road_block_reason:  str        = Form(
        ...,
        description=(
            "MANDATORY — reason for the road block, e.g. 'vehicle_breakdown', "
            "'accident', 'vip_movement', 'construction', 'public_event', "
            "'procession', 'protest', 'pot_holes', 'water_logging', 'tree_fall', "
            "'road_conditions', 'congestion', 'fog_low_visibility', 'debris', "
            "'others'. See GET /event-causes for the full list. Used to derive "
            "whether the event is planned or unplanned."
        ),
    ),
    latitude:  float = Form(12.9716,        description="Incident latitude (Bengaluru default)"),
    longitude: float = Form(77.5946,        description="Incident longitude (Bengaluru default)"),
    zone:      str   = Form("East Zone 1",  description="Traffic zone name"),
    corridor:  str   = Form("Non-corridor", description="Road corridor"),
    junction:  str   = Form(None,           description="Junction name (optional)"),
):
    """
    Main endpoint — upload accident image + location context + mandatory
    road block reason.

    Flow:
      1. YOLO object detection (sequential — feeds the rest)
      2. Resolve road_block_reason → canonical event_cause + planned/unplanned
      3. Road closure + disruption called in parallel
      4. Police personnel requirement estimated from congestion + emergency level
      5. All results merged into one dashboard-ready response

    For diversion routing call the traffic routing API independently:
      POST /api/routes/local-bypass  { accident_lat, accident_lon }
    """
    if not road_block_reason or not road_block_reason.strip():
        raise HTTPException(status_code=422, detail="road_block_reason is mandatory")

    event_cause, event_type = _resolve_event_cause(road_block_reason)

    image_bytes  = await file.read()
    filename     = file.filename or "upload.jpg"
    content_type = file.content_type or "image/jpeg"

    async with httpx.AsyncClient() as client:

        # ── Step 1: Object detection ───────────────────────────────────────
        try:
            raw_detection = await call_object_detection(
                client, image_bytes, filename, content_type
            )
            analysis = raw_detection.get("analysis", raw_detection)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Object detection API failed: {exc}\n{traceback.format_exc()}",
            )

        # ── Step 2: Build downstream payloads ─────────────────────────────
        closure_payload = _build_closure_payload(
            analysis, latitude, longitude, zone, corridor, junction,
            event_cause, event_type,
        )
        disruption_payload = _build_disruption_payload(analysis, closure_payload)

        # ── Step 3: Fan out in parallel ────────────────────────────────────
        closure_result, disruption_result = await asyncio.gather(
            asyncio.create_task(call_road_closure(client, closure_payload)),
            asyncio.create_task(call_disruption(client, disruption_payload)),
            return_exceptions=True,
        )

    # Wrap any exceptions
    closure_result    = _safe("road_closure", closure_result)
    disruption_result = _safe("disruption",   disruption_result)

    # ── Step 4: Parse each API response ────────────────────────────────────
    road_closure_req, closure_conf, closure_risk = _parse_closure(closure_result)
    severity_label, severity_conf, eta_range      = _parse_disruption(disruption_result)

    # ── Step 5: Shared fields from YOLO ────────────────────────────────────
    emergency        = analysis.get("emergency", {})
    em_level         = emergency.get("level", "LOW")
    scene_type       = emergency.get("scene_type", "unknown")
    congestion_score = float(analysis.get("congestion_score", 0.0))
    people           = int(analysis.get("people", 0))
    vehicles         = int(analysis.get("vehicles", 0))

    # ── Step 6: Derived outputs ────────────────────────────────────────────
    heatmap_points  = _generate_heatmap_points(latitude, longitude, congestion_score)
    recommendations = _generate_recommendations(
        em_level, road_closure_req, severity_label,
        people, vehicles, congestion_score,
    )
    police_estimate = _police_personnel_required(
        congestion_score, em_level, event_type, vehicles, people,
    )

    nearest_station = None
    if police_estimate["personnel_required"] >= POLICE_STATION_LOOKUP_THRESHOLD:
        async with httpx.AsyncClient() as client:
            nearest_station = await _find_nearest_police_station(client, latitude, longitude)

    # ── Step 7: Build response ─────────────────────────────────────────────
    return JSONResponse({
        "success":   True,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "incident": {
            "latitude":          latitude,
            "longitude":         longitude,
            "zone":              zone,
            "corridor":          corridor,
            "junction":          junction,
            "road_block_reason": road_block_reason,
            "event_cause":       event_cause,
            "event_type":        event_type,
        },

        # Panel 1 — Detected objects
        "detected_objects": {
            "total":           analysis.get("total_objects", 0),
            "vehicles":        vehicles,
            "people":          people,
            "road_blocks":     analysis.get("road_blocks", 0),
            "illegal_parking": analysis.get("illegal_parking", 0),
            "vehicle_types":   analysis.get("vehicle_types", {}),
        },

        # Panel 2 — Congestion & emergency
        "congestion": {
            "level":             analysis.get("congestion_level", "Unknown"),
            "score":             round(congestion_score, 3),
            "percentage":        _congestion_pct(congestion_score),
            "emergency_level":   em_level,
            "scene_type":        scene_type,
            "emergency_reasons": emergency.get("reasons", []),
        },

        # Panel 3 — Predictions
        "predictions": {
            "road_closure_required": road_closure_req,
            "closure_confidence":    closure_conf,
            "closure_risk_level":    closure_risk,
            "disruption_severity":   severity_label,
            "disruption_confidence": severity_conf,
            "expected_delay":        eta_range,
        },

        # Panel 4 — Map
        "map": {
            "heatmap_points": heatmap_points,
        },

        # Panel 5 — Dispatch / staffing
        "dispatch": {
            "police_personnel_required": police_estimate["personnel_required"],
            "personnel_bracket":         police_estimate["bracket"],
            "personnel_breakdown":       police_estimate["breakdown"],
            "station_lookup_threshold":  POLICE_STATION_LOOKUP_THRESHOLD,
            "nearest_police_station":    nearest_station,
        },

        # Panel 6 — Recommendations
        "recommendations": recommendations,

        # Raw API responses for debugging
        "_raw": {
            "object_detection": analysis,
            "road_closure":     closure_result,
            "disruption":       disruption_result,
        },
    })


@app.post("/analyze/demo", tags=["Analysis"])
async def analyze_demo(
    road_block_reason: str   = Form("accident", description="MANDATORY — reason for the road block"),
    latitude:          float = Form(12.9716),
    longitude:         float = Form(77.5946),
    zone:              str   = Form("East Zone 1"),
    corridor:          str   = Form("ORR East 1"),
):
    """
    Returns a realistic mocked response without calling any external APIs.
    Useful for frontend development before all services are running.
    """
    if not road_block_reason or not road_block_reason.strip():
        raise HTTPException(status_code=422, detail="road_block_reason is mandatory")

    event_cause, event_type = _resolve_event_cause(road_block_reason)

    congestion_score = 0.82
    heatmap_points   = _generate_heatmap_points(latitude, longitude, congestion_score)

    recommendations = _generate_recommendations(
        "CRITICAL", True, "90–240 mins (Major)", 8, 3, congestion_score
    )
    police_estimate = _police_personnel_required(
        congestion_score, "CRITICAL", event_type, vehicles=3, people=8,
    )

    return JSONResponse({
        "success":   True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incident":  {
            "latitude": latitude, "longitude": longitude,
            "zone": zone, "corridor": corridor, "junction": None,
            "road_block_reason": road_block_reason,
            "event_cause":       event_cause,
            "event_type":        event_type,
        },
        "detected_objects": {
            "total": 11, "vehicles": 3, "people": 8,
            "road_blocks": 0, "illegal_parking": 0,
            "vehicle_types": {"car": 1, "truck": 2},
        },
        "congestion": {
            "level": "Severe", "score": congestion_score, "percentage": 82,
            "emergency_level": "CRITICAL", "scene_type": "accident",
            "emergency_reasons": [
                "2 vehicle(s) in abnormally close proximity",
                "8 people clustered around vehicle(s)",
            ],
        },
        "predictions": {
            "road_closure_required": True,
            "closure_confidence":    0.874,
            "closure_risk_level":    "High",
            "disruption_severity":   "90–240 mins (Major)",
            "disruption_confidence": "81.2%",
            "expected_delay":        "90 – 240 mins",
        },
        "map": {
            "heatmap_points": heatmap_points,
        },
        "dispatch": {
            "police_personnel_required": police_estimate["personnel_required"],
            "personnel_bracket":         police_estimate["bracket"],
            "personnel_breakdown":       police_estimate["breakdown"],
            "station_lookup_threshold":  POLICE_STATION_LOOKUP_THRESHOLD,
            "nearest_police_station": (
                {
                    "name":               "Demo Traffic Police Station",
                    "latitude":           latitude + 0.004,
                    "longitude":          longitude - 0.003,
                    "approx_distance_km": 0.52,
                    "address":            "Demo Road, " + zone,
                    "phone":              None,
                    "source":             "mocked — demo mode",
                }
                if police_estimate["personnel_required"] >= POLICE_STATION_LOOKUP_THRESHOLD
                else None
            ),
        },
        "recommendations": recommendations,
        "_raw": {"note": "demo mode — no real API calls made"},
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8004, reload=True)