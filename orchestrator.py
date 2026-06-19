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
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CrowdFlow AI — Accident Analysis Orchestrator",
    description=(
        "Uploads an accident image, calls all specialist APIs in parallel, "
        "and returns a single merged payload ready for the dashboard. "
        "Routing is independent — call the traffic routing API directly."
    ),
    version="2.1.0",
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
) -> dict:
    """Build the exact payload expected by road_closure /predict."""
    emergency = analysis.get("emergency", {})
    scene     = emergency.get("scene_type", "unknown")
    em_level  = emergency.get("level", "LOW")
    veh_types = analysis.get("vehicle_types", {})

    cause_map = {
        "accident":  "accident",
        "breakdown": "vehicle_breakdown",
        "fire":      "others",
        "flood":     "water_logging",
    }
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
        "event_cause":    cause_map.get(scene, "others"),
        "priority":       priority_map.get(em_level, "Low"),
        "zone":           zone,
        "corridor":       corridor,
        "event_type":     "unplanned",
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
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "CrowdFlow AI Orchestrator v2.1"}


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


@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file:      UploadFile = File(...,            description="Accident scene image"),
    latitude:  float      = Form(12.9716,        description="Incident latitude (Bengaluru default)"),
    longitude: float      = Form(77.5946,        description="Incident longitude (Bengaluru default)"),
    zone:      str        = Form("East Zone 1",  description="Traffic zone name"),
    corridor:  str        = Form("Non-corridor", description="Road corridor"),
    junction:  str        = Form(None,           description="Junction name (optional)"),
):
    """
    Main endpoint — upload accident image + location context.

    Flow:
      1. YOLO object detection (sequential — feeds the rest)
      2. Road closure + disruption called in parallel
      3. All results merged into one dashboard-ready response

    For diversion routing call the traffic routing API independently:
      POST /api/routes/local-bypass  { accident_lat, accident_lon }
    """
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
        closure_payload    = _build_closure_payload(
            analysis, latitude, longitude, zone, corridor, junction
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

    # ── Step 7: Build response ─────────────────────────────────────────────
    return JSONResponse({
        "success":   True,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "incident": {
            "latitude":  latitude,
            "longitude": longitude,
            "zone":      zone,
            "corridor":  corridor,
            "junction":  junction,
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

        # Panel 5 — Recommendations
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
    latitude:  float = Form(12.9716),
    longitude: float = Form(77.5946),
    zone:      str   = Form("East Zone 1"),
    corridor:  str   = Form("ORR East 1"),
):
    """
    Returns a realistic mocked response without calling any external APIs.
    Useful for frontend development before all services are running.
    """
    congestion_score = 0.82
    heatmap_points   = _generate_heatmap_points(latitude, longitude, congestion_score)

    recommendations = _generate_recommendations(
        "CRITICAL", True, "90–240 mins (Major)", 8, 3, congestion_score
    )

    return JSONResponse({
        "success":   True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "incident":  {"latitude": latitude, "longitude": longitude,
                      "zone": zone, "corridor": corridor, "junction": None},
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
        "recommendations": recommendations,
        "_raw": {"note": "demo mode — no real API calls made"},
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8004, reload=True)