import os, re, base64
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Umbrales iniciales: AJUSTAR con tus pruebas reales
UMBRAL_COINCIDE = int(os.getenv("UMBRAL_COINCIDE", 100))
UMBRAL_NO_COINCIDE = int(os.getenv("UMBRAL_NO_COINCIDE", 30))
TOL_COLOR = 12
RATIO = 0.8
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
REF_DIR = os.path.join(os.path.dirname(__file__), "referencias")

app = FastAPI()
ANCHO = int(os.getenv("ANCHO", 1280))
sift = cv2.SIFT_create(int(os.getenv("N_PUNTOS", 4000)))
bf = cv2.BFMatcher()
_cache = {}


def preparar(img, ancho=None):
    ancho = ancho or ANCHO
    h, w = img.shape[:2]
    img = cv2.resize(img, (ancho, int(h * ancho / w)), interpolation=cv2.INTER_AREA)
    m = img.reshape(-1, 3).mean(0)
    return np.clip(img * (m.mean() / (m + 1e-6)), 0, 255).astype(np.uint8)


def rootsift(d):
    d = d / (d.sum(1, keepdims=True) + 1e-7)
    return np.sqrt(d).astype(np.float32)


def analizar(img):
    img = preparar(img)
    kp, d = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    return {"img": img, "kp": kp, "d": d, "lab": cv2.cvtColor(img, cv2.COLOR_BGR2LAB)}


def color_local(lab, x, y, r=3):
    return lab[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1].reshape(-1, 3).mean(0)


def comparar(A, B):
    if A["d"] is None or B["d"] is None or len(A["kp"]) < 8 or len(B["kp"]) < 8:
        return 0, []
    pares = bf.knnMatch(rootsift(A["d"]), rootsift(B["d"]), k=2)
    buenos = [p[0] for p in pares if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(buenos) < 8:
        return 0, []
    a = np.float32([A["kp"][m.queryIdx].pt for m in buenos])
    b = np.float32([B["kp"][m.trainIdx].pt for m in buenos])
    H, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 3.0)
    if H is None:
        return 0, []
    ok = []
    for m, inl in zip(buenos, mask.ravel()):
        if not inl:
            continue
        x1, y1 = map(int, A["kp"][m.queryIdx].pt)
        x2, y2 = map(int, B["kp"][m.trainIdx].pt)
        c1, c2 = color_local(A["lab"], x1, y1), color_local(B["lab"], x2, y2)
        if np.linalg.norm(c1[1:] - c2[1:]) < TOL_COLOR:
            ok.append(m)
    return len(ok), ok


def cargar_refs(oid):
    if oid in _cache:
        return _cache[oid]
    carpeta = os.path.join(REF_DIR, oid)
    if not os.path.isdir(carpeta):
        return None
    refs = {}
    for f in sorted(os.listdir(carpeta)):
        if f.lower().endswith(EXT):
            img = cv2.imread(os.path.join(carpeta, f))
            if img is not None:
                refs[f] = analizar(img)
    _cache[oid] = refs
    return refs


def valido(oid):
    return re.fullmatch(r"[A-Za-z0-9_-]{1,64}", oid) is not None


@app.get("/api/objeto/{oid}")
def info(oid: str):
    if not valido(oid) or not os.path.isdir(os.path.join(REF_DIR, oid)):
        raise HTTPException(404, "Objeto no encontrado")
    files = [f for f in sorted(os.listdir(os.path.join(REF_DIR, oid))) if f.lower().endswith(EXT)]
    return {"id": oid, "imagenes": [f"/ref/{oid}/{f}" for f in files]}


@app.post("/api/verificar")
async def verificar(id: str = Form(...), foto: UploadFile = File(...)):
    if not valido(id):
        raise HTTPException(400, "ID inválido")
    refs = cargar_refs(id)
    if not refs:
        raise HTTPException(404, "Objeto no encontrado")
    datos = np.frombuffer(await foto.read(), np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "No se pudo leer la imagen")
    F = analizar(img)
    if F["d"] is None or len(F["kp"]) < 50:
        return {"veredicto": "mala_foto", "puntaje": 0,
                "mensaje": "La foto no tiene suficiente detalle. Acercate, enfocá y evitá reflejos."}
    mejor, mref, mm = -1, None, []
    for n, R in refs.items():
        s, ms = comparar(F, R)
        if s > mejor:
            mejor, mref, mm = s, n, ms
    if mejor >= UMBRAL_COINCIDE:
        v = "coincide"
    elif mejor < UMBRAL_NO_COINCIDE:
        v = "no_coincide"
    else:
        v = "dudoso"
    imagen = None
    if mm:
        vis = cv2.drawMatches(F["img"], F["kp"], refs[mref]["img"], refs[mref]["kp"], mm[:120], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        vis = cv2.resize(vis, None, fx=0.4, fy=0.4)
        imagen = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", vis)[1]).decode()
    return {"veredicto": v, "puntaje": int(mejor), "referencia": mref, "coincidencias": imagen}


app.mount("/ref", StaticFiles(directory=REF_DIR), name="ref")


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
