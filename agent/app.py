import os
import sys
import json
from datetime import datetime, date, timedelta
from typing import Any, Dict, List
import uuid

import requests
import streamlit as st

# ---------- Helpers & env config ----------

def env_or_secret(key: str, default: str | None = None) -> str | None:
    """Try st.secrets first, then env var."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# Make Streamlit bind correctly in Cloud Shell / Cloud Run
# (You can also pass these as CLI flags; this is a safe default.)
# Force Streamlit to bind to the correct Cloud Run port
PORT = int(os.environ.get("PORT", "8080"))
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_PORT"] = str(PORT)

# ---------- Optional Google auth handling ----------
# In Colab: do explicit user auth. In Cloud Run / Cloud Shell: rely on ADC.
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    try:
        from google.colab import auth as colab_auth  # type: ignore
        colab_auth.authenticate_user()  # no project_id arg in the real API
    except Exception as e:
        st.warning(f"Colab auth could not run: {e}")

# ---------- Vertex AI / LangChain setup ----------
try:
    import vertexai
    from vertexai.preview.reasoning_engines import LangchainAgent
    from langchain.tools import tool
    from langchain_google_vertexai import (
        VertexAI as LCVertexAI,
        HarmBlockThreshold,
        HarmCategory,
    )
except ImportError as e:
    st.error(
        "Missing required libraries (vertexai, langchain_google_vertexai). "
        "Install the requirements listed below."
    )
    st.stop()

PROJECT_ID = env_or_secret("GCP_PROJECT_ID")
LOCATION = env_or_secret("GCP_LOCATION", "us-central1")
STAGING_BUCKET = env_or_secret("GCP_STAGING_BUCKET")

if not PROJECT_ID:
    st.error("GCP_PROJECT_ID is not configured (secret or env var).")
    st.stop()

try:
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
except Exception as e:
    st.error(f"Vertex AI init failed: {e}")
    st.stop()

MODEL_NAME = "gemini-2.5-flash"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
AGENT_EXECUTOR_KWARGS = {
    "return_intermediate_steps": True,
    "max_iterations": 5,
    "handle_parsing_errors": True,
}

# ---------- Tool helpers ----------
def _ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data}

def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}

# Tools
@tool
def get_user_interests(preferences_json: str) -> Dict[str, Any]:
    """Normalize and store user trip preferences from a JSON string."""
    try:
        prefs = json.loads(preferences_json)
        prefs["_normalized"] = True
        return _ok(prefs)
    except Exception as e:
        return _err(f"Invalid JSON: {e}")

@tool
def translate_text(text: str, target_lang: str = "en") -> Dict[str, Any]:
    """Translate arbitrary text to target_lang using Vertex AI via LangChain."""
    llm = LCVertexAI(
        model_name=MODEL_NAME,
        temperature=0.0,
        max_output_tokens=512,
        safety_settings=SAFETY_SETTINGS,
    )
    out = llm.invoke(f"Translate to {target_lang} preserving meaning and tone. Text:\n{text}")
    return _ok({"translated": out})

@tool
def get_weather(location: str, trip_date: str) -> Dict[str, Any]:
    """Get daily forecast (tmax/tmin/precip prob) for location on YYYY-MM-DD via Open-Meteo."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return _err("Location not found")
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        fc = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_mean",
                ],
                "timezone": "auto",
            },
            timeout=10,
        ).json()
        daily = fc.get("daily", {})
        if not daily:
            return _err("No forecast data")
        dates = daily.get("time", [])
        if trip_date not in dates:
            return _err("Requested date outside forecast range")
        i = dates.index(trip_date)
        precip_series = daily.get("precipitation_probability_mean") or []
        precip_val = precip_series[i] if i < len(precip_series) else None
        return _ok(
            {
                "date": trip_date,
                "tmax": daily["temperature_2m_max"][i],
                "tmin": daily["temperature_2m_min"][i],
                "precip_prob": precip_val,
            }
        )
    except Exception as e:
        return _err(str(e))

@tool
def get_popular_places(location: str, max_results: int = 10) -> Dict[str, Any]:
    """Return a stub list of popular places for a city (replace with Google Places)."""
    sample = [
        {"name": "Baga Beach", "type": "Beach", "cost": "free", "best_time": "evening"},
        {"name": "Fort Aguada", "type": "Fort", "cost": 50, "best_time": "sunset"},
        {"name": "Basilica of Bom Jesus", "type": "Church", "cost": 0, "best_time": "morning"},
        {"name": "Anjuna Flea Market", "type": "Market", "cost": 0, "best_time": "Wednesday"},
    ]
    return _ok(sample[:max_results])

@tool
def get_events_info(location: str, start: str, end: str, max_results: int = 10) -> Dict[str, Any]:
    """Return a stub list of events between dates (replace with Ticketmaster/Meetup/etc.)."""
    return _ok(
        [
            {"title": "Beach Music Night", "date": start, "venue": "Baga Beach"},
            {"title": "Goan Food Fest", "date": end, "venue": "Candolim"},
        ][:max_results]
    )

@tool
def search_hotels(area: str, budget_per_night: int = 5000, rooms: int = 1, nights: int = 3) -> Dict[str, Any]:
    """Return stub hotel search results filtered by budget; compute total by nights."""
    base = [
        {"name": "Sea Breeze Resort", "area": area, "price": 4200, "rating": 4.2},
        {"name": "Palm Grove Inn", "area": area, "price": 5600, "rating": 4.4},
        {"name": "Lagoon Stay", "area": area, "price": 3500, "rating": 4.0},
    ]
    fit = [h for h in base if h["price"] <= budget_per_night]
    total = [dict(h, total_cost=h["price"] * nights * rooms) for h in fit]
    return _ok(total)

@tool
def transport_options(source: str, dest: str, date_str: str) -> Dict[str, Any]:
    """Return stub flight/train/bus options between source and dest on date_str."""
    return _ok(
        [
            {"mode": "flight", "carrier": "IndiGo", "depart": f"{date_str} 08:15", "arrive": f"{date_str} 10:00", "price": 4500},
            {"mode": "train", "number": "22951", "depart": f"{date_str} 18:30", "arrive": f"{date_str} 06:45+1", "price": 1200},
            {"mode": "bus", "operator": "VRL", "depart": f"{date_str} 21:00", "arrive": f"{date_str} 08:00+1", "price": 900},
        ]
    )

@tool
def optimize_itinerary(day_plan_json: str) -> Dict[str, Any]:
    """Greedy itinerary optimizer: sort by priority desc, then end time, add 30m buffers."""
    try:
        items = json.loads(day_plan_json)
        items = sorted(
            items,
            key=lambda x: (-int(x.get("priority", 3)), x.get("end", "23:59"), int(x.get("est_minutes", 60))),
        )
        cur = datetime.strptime(items[0].get("start", "09:00"), "%H:%M") if items else None
        for it in items:
            if cur is None:
                cur = datetime.strptime(it.get("start", "09:00"), "%H:%M")
            it["scheduled_start"] = cur.strftime("%H:%M")
            cur = cur + timedelta(minutes=int(it.get("est_minutes", 60)) + 30)
        return _ok(items)
    except Exception as e:
        return _err(f"Optimization input error: {e}")

@tool
def social_media_suggestions(location: str) -> Dict[str, Any]:
    """Suggest photo spots and hashtags for the given location (stub)."""
    return _ok(
        {
            "photo_spots": ["Fort Aguada viewpoint", "Candolim beach dunes", "Fontainhas street art"],
            "hashtags": ["#GoaDiaries", "#SunsetVibes", "#TravelWithAI"],
        }
    )

@tool
def save_trip_plan(plan_json: str) -> Dict[str, Any]:
    """Save trip plan to JSON/HTML; PDF if WeasyPrint is available."""
    obj = plan_json
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            obj = {"raw": plan_json}
    if isinstance(obj, list):
        plan = {"id": str(uuid.uuid4()), "overview": "Auto-wrapped from list", "days": obj}
    elif isinstance(obj, dict):
        plan = obj
        plan.setdefault("id", str(uuid.uuid4()))
    else:
        plan = {"id": str(uuid.uuid4()), "raw": obj}

    base = f"trip_plan_{plan['id']}"
    json_path, html_path = f"{base}.json", f"{base}.html"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    html = "<html><body><h1>Trip Plan</h1><pre>" + json.dumps(plan, indent=2, ensure_ascii=False) + "</pre></body></html>"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    pdf_exists = False
    try:
        from weasyprint import HTML  # optional
        pdf_path = f"{base}.pdf"
        HTML(string=html).write_pdf(pdf_path)
        pdf_exists = True
    except Exception:
        pass

    out = {"json": os.path.abspath(json_path), "html": os.path.abspath(html_path)}
    if pdf_exists:
        out["pdf"] = os.path.abspath(pdf_path)
    return _ok(out)

@tool
def search_places_with_google_api(query: str, location: str | None = None, radius: int = 50000, place_type: str | None = None):
    """Search for places using Google Places API (stub)."""
    st.warning("This tool is a stub and requires a complete Google Places API implementation.")
    return []

@tool
def payment_checkout(amount_in_inr: int, reference: str = "TRIP-DEMO") -> Dict[str, Any]:
    """Mock payment; return pending status + a fake payment link."""
    return _ok({"status": "PENDING", "amount": amount_in_inr, "payment_link": f"https://example.com/pay/{reference}"})

@tool
def live_disruption_alerts(location: str, window_hours: int = 24) -> Dict[str, Any]:
    """Stub live alerts for disruptions; replace with real feeds later."""
    return _ok({"alerts": [f"No major disruptions expected in {location} next {window_hours}h."]})

# ------------------ UI ------------------
st.title("🗺️ AI-Powered Trip Planner")
st.markdown("**Please enter trip details **")

# Prompt Box with Generate Button Below
user_prompt = st.text_area(
    "Custom Prompt",
    placeholder="e.g., Plan a 3-day Goa trip with beaches and food..."
)

# Generate Button
send_prompt = st.button("🚀 Generate Trip Plan")

# Trip Details Sidebar
st.sidebar.header("Trip Details")
with st.sidebar.form(key="trip_form"):
    trip_type = st.selectbox("Trip Type*", ["family", "solo", "group", "couple"])
    budget = st.number_input("Budget (INR)*", min_value=1000, value=45000)
    people = st.number_input("Number of People*", min_value=1, value=3)
    children = st.text_input("Child Ages (e.g., '4, 10')")   # optional
    orig = st.text_input("Origin City*", value="Ahmedabad")
    dest = st.text_input("Destination City*", value="Goa")
    start_date = st.date_input("Start Date*", date.today())
    end_date = st.date_input("End Date*", date.today() + timedelta(days=2))
    diet = st.selectbox("Dietary Preference", ["vegetarian", "non-vegetarian", "vegan"])
    interests = st.multiselect(
        "Interests*",
        ["beach", "forts", "markets", "historical sites", "food"],
        ["beach", "forts", "markets"]
    )
    transport = st.selectbox("Mode of Transport*", ["flight", "train", "bus"])

    # 🔹 Language inside form (default English)
    language = st.selectbox(
        "Select Language",
        [
            "English (en)", "Hindi (hi)", "Bengali (bn)", "Gujarati (gu)", "Kannada (kn)",
            "Malayalam (ml)", "Marathi (mr)", "Manipuri (mni-Mtei)", "Nepali (ne)",
            "Odia (or)", "Punjabi (pa)", "Sindhi (sd)", "Tamil (ta)", "Telugu (te)",
            "Urdu (ur)", "Assamese (as)"
        ],
        index=0  # default English
    )

    submit_button = st.form_submit_button(label="Plan My Trip")


# ------------------ JSON Builder ------------------
def build_trip_request():
    child_ages = [{"age": int(a.strip())} for a in children.split(",") if a.strip().isdigit()]
    lang_code = language.split("(")[-1].replace(")", "").strip()  # extract e.g. "en"
    req = {
        "trip_type": trip_type,
        "budget": budget,
        "people": people,
        "dates": {"start": str(start_date), "end": str(end_date)},
        "orig": orig,
        "dest": dest,
        "children": child_ages,
        "diet": diet,
        "interests": interests,
        "Mode_of_transport": transport,
        "languages": [lang_code],
    }
    if user_prompt.strip():
        req["user_prompt"] = user_prompt.strip()
    return req
# Agent setup
agent = LangchainAgent(
    model=MODEL_NAME,
    tools=[search_hotels, get_popular_places, transport_options, get_events_info,
           get_weather, save_trip_plan, payment_checkout, live_disruption_alerts],
    agent_executor_kwargs=AGENT_EXECUTOR_KWARGS,
    system_instruction=(
            "You are a helpful multilingual travel planner. "
            "Please note that the 'get_distance_time' tool is unavailable. "
            # "Also, the 'search_places_with_google_api' tool is a stub and does not provide real data. "
            "Always create detailed day plans with exact timing. "
            "Ensure vegetarian/non-vegetarian food options based on the inputs. "
            "When budget and transport are specified, strictly follow them. "
            "Explain results in a structured, polite, and concise way."
        ),
)

# Handle actions
if send_prompt and user_prompt.strip():
    st.info("✈️ Generating trip plan from custom prompt...")
    trip_request = {"user_prompt": user_prompt.strip()}
    try:
        result = agent.query(input=f"Plan a trip. {json.dumps(trip_request)}")
        st.success("Trip plan generated!")
        st.markdown(result.get("output", "_No output from agent._"))
    except Exception as e:
        st.error(f"Error: {e}")

elif submit_button and user_prompt.strip():
    st.info("✈️ Generating trip plan from trip details + custom prompt...")
    trip_request = build_trip_request()
    try:
        result = agent.query(input=f"Plan a trip. {json.dumps(trip_request)}")
        st.success("Trip plan generated!")
        st.markdown(result.get("output", "_No output from agent._"))
    except Exception as e:
        st.error(f"Error: {e}")

elif submit_button:
    st.info("✈️ Generating trip plan from trip details...")
    trip_request = build_trip_request()
    try:
        result = agent.query(input=f"Plan a trip. {json.dumps(trip_request)}")
        st.success("Trip plan generated!")
        st.markdown(result.get("output", "_No output from agent._"))
    except Exception as e:
        st.error(f"Error: {e}")
