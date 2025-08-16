from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPHHOPPER_API_KEY = os.getenv("GRAPHHOPPER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")


timeout = httpx.Timeout(30.0, connect=10.0)
openrouter_lock = asyncio.Lock()

async def geocode(place: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        return float(data[0]["lat"]), float(data[0]["lon"])

async def get_mapillary_images(bbox: str, limit: int = 5):
    url = "https://graph.mapillary.com/images"
    params = {
        "bbox": bbox,
        "limit": limit,
        "fields": "id,computed_geometry,thumb_1024_url",
        "access_token": MAPILLARY_TOKEN,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

async def analyze_image(image_url: str, retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemma-3-27b-it:free",
        "messages": [
            {
                "role": "user",
                "content": f"Analyse cette image et réponds uniquement par 'safe' ou 'danger': {image_url}"
            }
        ],
    }

    async with openrouter_lock:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(retries):
                try:
                    resp = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    result = resp.json()
                    return result["choices"][0]["message"]["content"].strip().lower()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        retry_after = e.response.headers.get("Retry-After")
                        wait = int(retry_after) if retry_after else 5
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise HTTPException(status_code=429, detail="Trop de requêtes vers OpenRouter, réessayer plus tard.")

async def classify_route_safety(images):
    safe_count = 0
    danger_count = 0
    max_images = 5  # Limite nombre images analysées

    for img in images[:max_images]:
        image_url = img.get("thumb_1024_url")
        if not image_url:
            continue
        decision = await analyze_image(image_url)
        if "danger" in decision:
            danger_count += 1
        else:
            safe_count += 1
        await asyncio.sleep(1)  # Pause pour éviter rate-limit

    return "safe" if safe_count >= danger_count else "danger"

@app.get("/")
async def root():
    return {"message": "API fonctionne !"}



@app.get("/route")
async def get_route(start_place: str = Query(...), end_place: str = Query(...)):
    start_coords = await geocode(start_place)
    end_coords = await geocode(end_place)
    if not start_coords or not end_coords:
        return {"error": "Lieu introuvable"}

    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords

    gh_url = "https://graphhopper.com/api/1/route"
    params = {
        "point": [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
        "vehicle": "foot",
        "locale": "fr",
        "points_encoded": "false",
        "algorithm": "alternative_route",
        "alternative_route.max_paths": 3,
        "key": GRAPHHOPPER_API_KEY
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(gh_url, params=params)
        resp.raise_for_status()
        route_data = resp.json()

    routes_with_status = []

    for path in route_data.get("paths", []):
        coords_list = path["points"]["coordinates"]  # [ [lon, lat], ... ]
        lons = [c[0] for c in coords_list]
        lats = [c[1] for c in coords_list]

        # Création bbox un peu élargie autour itinéraire
        bbox = f"{min(lons)-0.005},{min(lats)-0.005},{max(lons)+0.005},{max(lats)+0.005}"

        # Récupérer images Mapillary dans bbox
        images_data = await get_mapillary_images(bbox, limit=10)
        images = images_data.get("data", [])

        # Analyser les images
        status = await classify_route_safety(images)

        routes_with_status.append({
            "geometry": path["points"],
            "distance": path.get("distance"),
            "time": path.get("time"),
            "status": status,
        })

    return {
        "routes": routes_with_status,
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
    }
