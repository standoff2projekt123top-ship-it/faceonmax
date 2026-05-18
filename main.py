
import os, math, uuid, cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
mp_face = mp.solutions.face_mesh

IDEAL = {
    "face_ratio": 1.38,
    "compactness": 0.72,
    "eye_spacing": 0.46,
    "eye_area": 0.30,
    "lower_third": 0.34,
    "jaw_width": 0.76,
    "cheekbone_to_jaw": 1.18,
    "vertical_balance": 1.00,
    "symmetry": 1.00,
}

def clamp(v, lo=1, hi=10):
    return max(lo, min(hi, v))

def score_distance(v, ideal, tol):
    return round(clamp(10 - abs(v - ideal) / tol * 5), 1)

def score_min(v, good, tol):
    return round(clamp(10 - max(0, good - v) / tol * 5), 1)

def dist(a,b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def subtier(score, name):
    d = score - math.floor(score)
    return ("L" if d < .34 else "M" if d < .67 else "H") + name

def tier(score):
    if score < 2: return "SUB HUMAN"
    if score < 3: return "SUB3"
    if score < 4: return "SUB5"
    if score < 5: return subtier(score, "LTN")
    if score < 6: return subtier(score, "MTN")
    if score < 7: return subtier(score, "HTN")
    if score < 8: return "CHAD LITE"
    if score < 9: return "CHAD"
    if score < 10: return "ADAM LITE"
    return "TRUE ADAM"

def top_percent(score):
    m = {1:99.9,2:96,3:82,4:60,5:35,6:15,7:7,8:2,9:.3,10:.01}
    lo = int(clamp(score)); hi = min(10, lo+1)
    if lo == hi: return m[lo]
    return round(m[lo] + (m[hi]-m[lo])*(score-lo), 2)

def pt(lm, idx, w, h):
    return int(lm[idx].x*w), int(lm[idx].y*h)

def save_file(file, prefix):
    path = os.path.join(UPLOAD_DIR, f"{prefix}_{uuid.uuid4().hex}.jpg")
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path

def get_landmarks(path):
    img = cv2.imread(path)
    if img is None:
        return None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return img, None
    return img, res.multi_face_landmarks[0].landmark

def face_bbox(lm, w, h, pad=0.22):
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x1, x2 = max(0, min(xs)), min(1, max(xs))
    y1, y2 = max(0, min(ys)), min(1, max(ys))
    bw, bh = x2-x1, y2-y1
    x1 = max(0, x1 - bw*pad); x2 = min(1, x2 + bw*pad)
    y1 = max(0, y1 - bh*pad); y2 = min(1, y2 + bh*pad)
    return int(x1*w), int(y1*h), int(x2*w), int(y2*h)

def draw_lines(path, lm, kind):
    img = cv2.imread(path)
    h,w = img.shape[:2]
    x1,y1,x2,y2 = face_bbox(lm,w,h)
    crop = img[y1:y2, x1:x2].copy()
    ch,cw = crop.shape[:2]

    green=(0,255,130); gold=(56,145,202)

    def local(idx):
        x,y = pt(lm,idx,w,h)
        return int(x-x1), int(y-y1)
    def line(a,b,c=green,t=3):
        cv2.line(crop, local(a), local(b), c, t)
    def dot(i,c=green):
        cv2.circle(crop, local(i), max(4,cw//160), c, -1)

    groups = {
        "hunter": [(33,133),(362,263),(159,145),(386,374)],
        "jaw_projection": [(172,152),(397,152),(2,152)],
        "profile": [(10,1),(1,152),(10,152)],
        "dimorphism": [(172,397),(234,454),(2,152)],
        "orbital": [(33,263),(159,386),(145,374)],
        "cheekbones": [(234,454),(172,397)],
        "compactness": [(234,454),(10,152)],
        "eye_area": [(33,133),(362,263),(159,145),(386,374)],
        "lower_third": [(2,152),(61,291),(172,152),(397,152)],
        "ramus": [(172,152),(397,152),(172,397)],
    }
    for n,(a,b) in enumerate(groups.get(kind, [(10,152)])):
        line(a,b,gold if n==0 else green,3)
        dot(a); dot(b)

    cv2.rectangle(crop, (6,6), (cw-6,ch-6), green, 2)
    cv2.putText(crop, kind.upper(), (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, gold, 2)

    out = os.path.join(UPLOAD_DIR, f"{kind}_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(out, crop)
    return "/" + out.replace("\\","/")

def analyze_front(path):
    img,lm = get_landmarks(path)
    if lm is None: return None

    lf,rf = lm[234],lm[454]
    forehead,chin = lm[10],lm[152]
    lo,ro = lm[33],lm[263]
    li,ri = lm[133],lm[362]
    lt,lb = lm[159],lm[145]
    rt,rb = lm[386],lm[374]
    ln,rn = lm[129],lm[358]
    nt,nb = lm[1],lm[2]
    mouth_l,mouth_r = lm[61],lm[291]
    lj,rj = lm[172],lm[397]

    face_w = dist(lf,rf)
    face_h = dist(forehead,chin)

    face_ratio = face_h / max(face_w,.001)
    compact = face_w / max(face_h,.001)
    eye_spacing = dist(lo,ro) / max(face_w,.001)
    eye_area = ((dist(lt,lb)/max(dist(lo,li),.001)) + (dist(rt,rb)/max(dist(ro,ri),.001))) / 2
    nose_width = dist(ln,rn) / max(face_w,.001)
    mouth_width = dist(mouth_l,mouth_r) / max(face_w,.001)
    lower = dist(nb,chin) / max(face_h,.001)
    vertical = dist(forehead,nb) / max(dist(nb,chin),.001)
    jaw_width = dist(lj,rj) / max(face_w,.001)
    cheek_jaw = face_w / max(dist(lj,rj),.001)
    canthal = ((li.y-lo.y)+(ro.y-ri.y))/2

    axis = (forehead.x+chin.x+nt.x)/3
    diffs=[]
    for l,r in [(lf,rf),(lo,ro),(li,ri),(ln,rn),(mouth_l,mouth_r),(lj,rj)]:
        dl=abs(l.x-axis); dr=abs(r.x-axis)
        if max(dl,dr)>0:
            diffs.append(abs(dl-dr)/max(dl,dr))
    symmetry = max(0,min(1,1-sum(diffs)/len(diffs)))

    face_s = score_distance(face_ratio, IDEAL["face_ratio"], .35)
    compact_s = score_distance(compact, IDEAL["compactness"], .18)
    eye_spacing_s = score_distance(eye_spacing, IDEAL["eye_spacing"], .18)
    eye_area_s = score_distance(eye_area, IDEAL["eye_area"], .12)
    lower_s = score_distance(lower, IDEAL["lower_third"], .12)
    vertical_s = score_distance(vertical, IDEAL["vertical_balance"], .35)
    jaw_s = score_distance(jaw_width, IDEAL["jaw_width"], .18)
    cheek_s = score_distance(cheek_jaw, IDEAL["cheekbone_to_jaw"], .25)
    sym_s = score_distance(symmetry, IDEAL["symmetry"], .30)

    hunter = round(clamp(eye_spacing_s*.20 + eye_area_s*.30 + sym_s*.10 + clamp(7.5+canthal*50)*.40),1)
    orbital = round((hunter + eye_area_s + eye_spacing_s) / 3,1)
    dimorphism = round((jaw_s + lower_s + compact_s + cheek_s) / 4,1)

    return {
        "lm": lm, "path": path,
        "scores": {
            "hunter": hunter,
            "orbital": orbital,
            "dimorphism": dimorphism,
            "cheekbones": cheek_s,
            "compactness": compact_s,
            "eye_area": eye_area_s,
            "lower_third": round((lower_s+jaw_s+vertical_s)/3,1),
        },
        "values": {
            "face_ratio": round(face_ratio,3),
            "compactness": round(compact,3),
            "eye_spacing": round(eye_spacing,3),
            "eye_area": round(eye_area,3),
            "nose_width": round(nose_width,3),
            "mouth_width": round(mouth_width,3),
            "lower_third": round(lower,3),
            "jaw_width": round(jaw_width,3),
            "cheekbone_to_jaw": round(cheek_jaw,3),
            "symmetry": round(symmetry,3),
            "canthal": round(canthal,4),
        }
    }

def analyze_profile(path):
    img,lm = get_landmarks(path)
    if lm is None: return None
    forehead,nose,nb,chin,jaw = lm[10],lm[1],lm[2],lm[152],lm[172]
    depth = abs(nose.x-jaw.x)
    chin_proj = abs(chin.x-nb.x)/max(depth,.001)
    profile_balance = abs(forehead.x-chin.x)
    ramus = abs(jaw.y-chin.y)

    jaw_projection = score_min(chin_proj,.28,.20)
    profile_score = round((jaw_projection + score_distance(profile_balance,.03,.18) + score_min(ramus,.16,.14))/3,1)
    ramus_score = score_min(ramus,.16,.14)

    return {
        "lm": lm, "path": path,
        "scores": {"jaw_projection":jaw_projection, "profile":profile_score, "ramus":ramus_score},
        "values": {"chin_projection":round(chin_proj,3), "profile_balance":round(profile_balance,3), "ramus":round(ramus,3)}
    }

@app.get("/")
def root():
    return {"status":"FaceOnMax face engine online"}

@app.post("/analyze")
async def analyze(front: UploadFile = File(...), profile: UploadFile = File(...)):
    front_path = save_file(front,"front")
    profile_path = save_file(profile,"profile")

    fr = analyze_front(front_path)
    if fr is None:
        return {"ok":False,"error":"Не нашёл лицо на фото анфас. Нужна чёткая фотка лица."}
    pr = analyze_profile(profile_path)
    if pr is None:
        return {"ok":False,"error":"Не нашёл лицо на фото профиль. Нужна чёткая боковая фотка."}

    raw = [
        ("hunter","ДЕТЕКТОР ХАНТЕР ГЛАЗ","front",fr["scores"]["hunter"],f"canthal={fr['values']['canthal']}, eye area={fr['values']['eye_area']}","45% кантал + 30% eye area + 20% eye spacing + 10% symmetry","Оценивает именно область глаз по landmark-точкам."),
        ("jaw_projection","ПРОЕКЦИЯ ЧЕЛЮСТИ","profile",pr["scores"]["jaw_projection"],f"chin projection={pr['values']['chin_projection']}","chin projection / face depth","Сравнивает подбородок и линию челюсти в профиле."),
        ("profile","АНАЛИЗ ПРОФИЛЯ","profile",pr["scores"]["profile"],f"profile balance={pr['values']['profile_balance']}","jaw projection + profile balance + ramus","Оценивает профиль по лбу, носу, подбородку и ramus."),
        ("dimorphism","ДИМОРФИЗМ","front",fr["scores"]["dimorphism"],f"jaw={fr['values']['jaw_width']}, compact={fr['values']['compactness']}","jaw width + lower third + compactness + cheekbones","Считает выраженность мужских черт по лицевым точкам."),
        ("orbital","ОРБИТАЛЬНЫЙ ВЕКТОР","front",fr["scores"]["orbital"],f"eye spacing={fr['values']['eye_spacing']}","hunter + eye area + eye spacing","Оценивает поддержку глазной зоны."),
        ("cheekbones","СКУЛЫ","front",fr["scores"]["cheekbones"],f"cheekbone/jaw={fr['values']['cheekbone_to_jaw']}","face width / jaw width","Считает выраженность скул относительно челюсти."),
        ("compactness","КОМПАКТНОСТЬ ЛИЦА","front",fr["scores"]["compactness"],f"compactness={fr['values']['compactness']}","face width / face height","Сравнивает ширину и высоту именно лица."),
        ("eye_area","ОБЛАСТЬ ГЛАЗ","front",fr["scores"]["eye_area"],f"eye area={fr['values']['eye_area']}","eye height / eye width","Считает форму глазной зоны."),
        ("lower_third","НИЖНЯЯ ТРЕТЬ","front",fr["scores"]["lower_third"],f"lower third={fr['values']['lower_third']}","lower third + jaw width + vertical balance","Оценивает подбородок, челюсть и нижнюю треть."),
        ("ramus","РАМУС","profile",pr["scores"]["ramus"],f"ramus={pr['values']['ramus']}","vertical ramus estimate","Оценивает ramus по профильным landmarks."),
    ]

    cats = []
    for key,title,photo,score,value,formula,desc in raw:
        src = fr if photo=="front" else pr
        cats.append({
            "key":key, "title":title, "photo":photo, "score":score,
            "value":value, "formula":formula, "desc":desc,
            "image_url":draw_lines(src["path"], src["lm"], key)
        })

    avg = round(sum(c["score"] for c in cats)/len(cats),2)
    potential = round(min(10,avg+.45),2)

    return {"ok":True, "psl_score":avg, "tier":tier(avg), "potential":potential,
            "potential_tier":tier(potential), "top_percent":top_percent(avg), "categories":cats}

if __name__ == "__main__":
    import uvicorn
    print("FaceOnMax backend: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
