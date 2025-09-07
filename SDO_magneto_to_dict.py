import os, glob, math
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.coordinates import sun
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.constants import R_sun
RSUN_MM = R_sun.to_value(u.Mm)


# ============== 설정 ==============
IN = r'D:\SDO\image\hmib'                 # 이미지 폴더
OUT_BASE = r'D:\SDO\image\hmib\mask_overlays'   # 출력 저장 루트
os.makedirs(OUT_BASE, exist_ok=True)

# ============== 공용 유틸 ==============
def gray(x): return cv.cvtColor(x, cv.COLOR_BGR2GRAY)
def as_unit(x): return x.astype(np.uint8)                  # 이진화할 때 (마스킹)
def as_float(x): return x.astype(np.float32)               # 계산할 때 (통계)
def zero_img(x): return np.zeros_like(x, dtype=np.uint8)   # 마스킹 바탕


def obstime_from_name(img_name: str):   # 사진 불러오기 
    """
    기대 형식: YYYYMMDD_HHMMSS*.jpg
    예: 20250819_172300_sdo.jpg
    """
    base_name = os.path.basename(img_name)
    splt = os.path.splitext(base_name)[0]
    yyyy = splt[0:4]; mm = splt[4:6]; dd = splt[6:8]
    hh   = splt[9:11]; mi = splt[11:13]; ss = splt[13:15]
    obstime = f"{yyyy}-{mm}-{dd} {hh}:{mi}:{ss}"
    return obstime



# ================================================
# ============== (1) 태양 원반 마스킹 ==============
# ================================================
def sundisk_mask_from_img(g):
    """Otsu 이진화로 전처리 후, 가장 큰 컨투어의 타원으로 태양disk 중심/반경 산출"""
    # 전처리 
    _, by = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    by = cv.morphologyEx(by, cv.MORPH_CLOSE, np.ones((3,3), np.uint8), 2)   # 구멍 메꾸기
    by = cv.medianBlur(by, 3) 

    # 가장 큰 컨투어 선택
    contours, _ = cv.findContours(by, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("태양 원반 컨투어를 찾지 못했습니다.")
    disk_cnt = max(contours, key=cv.contourArea)
    a_disk = cv.contourArea(disk_cnt)
    r_disk = math.sqrt(a_disk / np.pi)   # 면적 기반 반지름 pix

    # 타원 피팅 마스크 
    (cx, cy), (_, _), ang = cv.fitEllipse(disk_cnt)
    c_disk = (int(round(cx)), int(round(cy)))
    disk_mask = zero_img(g)
    cv.drawContours(disk_mask, [disk_cnt], -1, 255, -1)

    # 5픽셀 안쪽부터 침식(채우기) 
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))  # ≈ 3 px
    disk_mask_safe = cv.erode(disk_mask, kernel, iterations=1)

    return a_disk, r_disk, c_disk, disk_mask, disk_mask_safe



# ======================================================
# ============== (2) 자기장 극성 마스크 분리 ==============
# ======================================================
def magneto_mask_from_disk(g, disk_mask, disk_mask_safe):
    """간단히 임계값으로 구별"""
    pos_mask = zero_img(g)
    pos_mask[(disk_mask>0) & (g > 235)] = 255
    pos_mask = cv.bitwise_and(pos_mask, disk_mask_safe)

    neg_mask = zero_img(g)
    neg_mask[(disk_mask>0) & (g < 20)] = 255
    neg_mask = cv.bitwise_and(neg_mask, disk_mask_safe)

    return pos_mask, neg_mask


def save_masks(in_dir):
    """mask 이미지 저장"""
    OUT_DIRS = {"disk":  os.path.join(OUT_BASE, "disk_mask"),
                "positive": os.path.join(OUT_BASE, "pos_mask"),
                "negative": os.path.join(OUT_BASE, "neg_mask")}
    for d in OUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

    paths = glob.glob(os.path.join(in_dir, "*.jpg"))   # 폴더안 모든 jpg 경로 불러오기
    for p in paths:
        img = cv.imread(p, cv.IMREAD_COLOR)
        g = gray(img)
        base_name = os.path.splitext(os.path.basename(p))[0]   # 이미지 이름만 

        _, _, _, disk_mask, disk_mask_safe = sundisk_mask_from_img(g)
        pos_mask, neg_mask = magneto_mask_from_disk(g, disk_mask, disk_mask_safe)

        cv.imwrite(os.path.join(OUT_DIRS["disk"],  base_name + "_disk.jpg"),  disk_mask)
        cv.imwrite(os.path.join(OUT_DIRS["positive"], base_name + "_pos.jpg"), pos_mask)
        cv.imwrite(os.path.join(OUT_DIRS["negative"], base_name + "_neg.jpg"), neg_mask)
    print(f"✅ 완료: {len(paths)}장 처리됨")



# ===============================================
# ============== (3) 자기장 군집화 ==============
# ===============================================
def cluster_dbscan(mask):
    """각 마스크 따로 포인트를 추출해서 DBSCAN 군집화"""
    mask = cv.dilate(mask, cv.getStructuringElement(  # 살짝 팽창 (반암부 연결)
                              cv.MORPH_ELLIPSE,(5,5)), iterations=1)
    ys, xs = np.where(mask > 0)
    points = np.stack([xs, ys], axis=1) if len(xs) else np.empty((0,2), dtype=int)
    if len(points)==0: return np.array([]), points   # 군집 없을 때
    
    db = DBSCAN(eps=15, min_samples=300)
    labels = db.fit_predict(points)
    return labels, points


def props_from_cluster(g, labels, points, mask):
    """군집별 면적과 무게중심 좌표를 계산"""
    all_props = []
    if len(points)==0: return all_props   # 군집 없을 때
    for k in sorted([l for l in np.unique(labels) if l!=-1]):
        idx = np.where(labels==k)[0]
        if idx.size < 100:    # 점 개수 50개 이하면 스킵 
            continue
        pts = points[idx]

        # 군집 마스크 만들기
        cl_mask = zero_img(mask)
        cl_mask[pts[:,1], pts[:,0]] = 255
        cl_mask = cv.morphologyEx(cl_mask, cv.MORPH_CLOSE, 
                                  cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)),1)

        # 마스크 평균 밝기
        ys, xs = np.where((cl_mask>0) & (mask>0))   # 자기장 밝기 
        vals = g[ys, xs]
        mean_brightness = float(np.mean(vals)) if len(vals)>0 else 0.0
        sum_brightness  = float(np.sum(vals))  if len(vals)>0 else 0.0

        quiet_mask = (cl_mask>0) & (mask==0)        # 로컬 배경 밝기  
        vals_q = g[quiet_mask]
        mean_bright_q = float(np.mean(vals_q)) if len(vals_q)>0 else 0.0
        sum_bright_q  = float(np.sum(vals_q))  if len(vals_q)>0 else 0.0

        # 마스크 면적
        area = int(np.sum((cl_mask>0) & (mask>0)))

        # 마스크 무게중심
        m = cv.moments(cl_mask, binaryImage=True)
        cx = m['m10']/m['m00']; cy = m['m01']/m['m00']

        # 마스크 바운딩박스
        x1, y1 = int(pts[:,0].min()), int(pts[:,1].min())
        x2, y2 = int(pts[:,0].max()), int(pts[:,1].max())
        bbox = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

        all_props.append({"label": int(k), 
                          "centroid_px": (float(cx), float(cy)), "bbox": bbox,
                          "area_px": area,
                          "mean_bright": {"magneto": mean_brightness, "quiet": mean_bright_q},
                          "sum_bright": {"magneto": sum_brightness, "quiet": sum_bright_q},
                          "mask": cl_mask})
    return all_props


def add_helio_to_prop(obstime_str, a_disk, r_disk, c_disk, all_props):
    """원반 반경으로 arcsec/px 비율을 추출하여 헬리오 좌표로 변환, 보정 면적 계산
    (위성 이미지라 P 보정은 필요없다)"""
    obst = Time(obstime_str)
    ang_rad = sun.angular_radius(obst) 
    ang_arcsec = ang_rad.to(u.arcsec).value 
    scale = ang_arcsec / r_disk             # arcsec/pixel 
    alpha_rad = ang_rad.to_value(u.rad)   # 태양 각반경 α [rad]

    for p in all_props:
        # 지구에서 본 좌표 (projective)
        cx, cy = p["centroid_px"]        # [pix]
        dx = (cx - c_disk[0]) * scale    # [arcsec] 오른쪽(+)=서쪽
        dy = -(cy - c_disk[1]) * scale   # [arcsec] 위쪽(+)=북쪽
        hp = SkyCoord(Tx = dx*u.arcsec, Ty = dy*u.arcsec,   
                    frame=frames.Helioprojective(observer=get_horizons_coord('SDO', obst), obstime=obst))
        
        # 태양 좌표로 변환 (Stonyhurst)
        hg = hp.transform_to(frames.HeliographicStonyhurst(obstime=obst))
        LAT = float(hg.lat.to_value(u.deg))
        LON = float(hg.lon.to_value(u.deg))

        # 태양 기준 좌표로 변환 (Carrington)
        hgc = hp.transform_to(frames.HeliographicCarrington(obstime=obst))
        LON_CA = float(hgc.lon.to_value(u.deg) % 360.0)

        # foreshortening 보정 (μ=cos θ) 
        r_rad = float(np.hypot(dx, dy) * (u.arcsec).to(u.rad))   # 중심으로부터 각거리 r [rad]
        r_norm = min(1.0, r_rad/alpha_rad)                       # 0~1로 정규화
        mu = max(1e-6, float(np.sqrt(1.0 - r_norm**2)))          # 원반 가장자리(limb)에 갈수록 작아짐
        area_mu = p['area_px'] / mu

        # 관측되는 앞면 반구 전체를 백만으로 두는 단위 변환 (Millionths of Solar Hemisphere)
        MSH = float(area_mu / a_disk * 1e6/2)

        p["stonyhurst"] = {"lat": LAT, "lon": LON}
        p["carrington"] = LON_CA
        p["mu"] = mu
        p["area_MSH"] = MSH
    return all_props


def save_overlays(img, base_name, mask, all_props, pos_neg=''):
    """마스크 및 군집 바운딩박스 오버레이 이미지 저장"""
    OUT_DIR = os.path.join(OUT_BASE, f"overlays_{pos_neg}")
    os.makedirs(OUT_DIR, exist_ok=True)
    ov_img = img.copy()

    # 흑점 마스크 오버레이
    color = (255,0,0) if pos_neg=='pos' else (0,0,255)
    color_ov = np.array(color, dtype=np.uint8)
    ov_img[mask > 0] = as_unit(as_float(0.5*ov_img[mask > 0]) + 0.5*color_ov)

    for q in all_props:
        # mask에 딱 맞는 박스 그리기
        x, y, w, h = q["bbox"]
        cv.rectangle(ov_img, (x, y), (x + w, y + h), color, 2)

        # 무게중심 표시
        cx, cy = q["centroid_px"]
        cv.circle(ov_img, (int(round(cx)), int(round(cy))), 3, color, -1)

        txt = (f"H:{q['stonyhurst']['lat']:.1f}, "
               f"C:{q['carrington']:.1f}, "
               f"A:{q['area_px']}, ")
        cv.putText(ov_img, txt, (x, max(y-4, 10)), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    cv.imwrite(os.path.join(OUT_DIR, base_name + f"_overlay_{pos_neg}.jpg"), ov_img)



# ===============================================
# ============== (4) 자기장 최근접 쌍 ==============
# ===============================================
def great_circle_mm(latp, lonp, latn, lonn):
    """두 무게중심 사이 구면 거리 계산 = 각거리(rad) x 태양반지름(Mm)"""
    φ1, λ1, φ2, λ2 = map(np.deg2rad, [latp, lonp, latn, lonn])
    d = 2*np.arcsin(np.sqrt(np.sin((φ2-φ1)/2)**2 +
                            np.cos(φ1)*np.cos(φ2)*np.sin((λ2-λ1)/2)**2))
    return float(RSUN_MM * d)


def bipole_tilt_deg(latp, lonp, latn, lonn):
    """적도에 대한 기울기를 Joy's law(위도∝경사)로 계산"""
    φp, λp, φn, λn = map(np.deg2rad, [latp, lonp, latn, lonn])
    dlat = φn - φp
    dlon = np.arctan2(np.sin(λn - λp), np.cos(λn - λp))   # Δlon을 [-π, π] 범위로 안정화
    latm = 0.5 * (φp + φn)
    ang = np.arctan2(dlat, dlon*np.cos(latm) + 1e-12)
    return float(np.rad2deg(ang))


def match_bipoles(all_props_p, all_props_n):
    """최근접 매칭: 각 Pos에 대해 가장 가까운 Neg을 매칭"""
    pairs = []
    all_props_p = sorted(all_props_p, key=lambda p: p["area_px"], reverse=True)   # 면적 기준 정렬
    for pos in all_props_p:
        latp = pos["stonyhurst"]["lat"]; lonp = pos["stonyhurst"]["lon"]

        # 후보 N들 거리 계산
        candidate = []
        for j, neg in enumerate(all_props_n):
            latn = neg["stonyhurst"]["lat"]; lonn = neg["stonyhurst"]["lon"]
            gcm = great_circle_mm(latp, lonp, latn, lonn)  # 구면거리 (Mm)
            candidate.append((gcm, j))
        if not candidate:
            continue

        # 최단거리 후보 N 선택 
        gcm, jbest = min(candidate, key=lambda x: x[0])
        neg = all_props_n[jbest]
        tilt = bipole_tilt_deg(latp, lonp, neg["stonyhurst"]["lat"], neg["stonyhurst"]["lon"])
        pairs.append((pos, neg, gcm, tilt))
    return pairs


def save_bipole_overlay(img, base_name, pairs):
    OUT_DIR = os.path.join(OUT_BASE, "overlays")
    os.makedirs(OUT_DIR, exist_ok=True)
    ov = img.copy()

    for (pos, neg, gcm, tilt) in pairs:
        cxp, cyp = pos["centroid_px"]; cxn, cyn = neg["centroid_px"]
        xyp = (int(round(cxp)), int(round(cyp)))
        xyn = (int(round(cxn)), int(round(cyn)))

        # 연결선 & 중심점
        cv.line(ov, xyp, xyn, (0,255,255), 2)   # 노란 선
        cv.circle(ov, xyp, 4, (255,0,0), -1)    # P: 파란 점
        cv.circle(ov, xyn, 4, (0,0,255), -1)    # N: 빨강 점

        # 텍스트 (분리/기울기)
        mid = ( (xyp[0]+xyn[0])//2, (xyp[1]+xyn[1])//2 )
        txt = f"gcm={gcm:.0f}Mm  tilt={tilt:+.1f}°"
        cv.putText(ov, txt, (mid[0]+6, mid[1]-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv.LINE_AA)
    cv.imwrite(os.path.join(OUT_DIR, base_name + "_overlay_pairs.jpg"), ov)



# ==================================
# ============== 메인 ==============
# ==================================
def process_folder(in_dir):
    paths = glob.glob(os.path.join(in_dir, "*.jpg"))
    for pth in paths:
        img = cv.imread(pth, cv.IMREAD_COLOR)
        g = gray(img)
        base_name = os.path.splitext(os.path.basename(pth))[0]
        obstime = obstime_from_name(pth)

        a_disk, r_disk, c_disk, disk_mask, disk_mask_safe = sundisk_mask_from_img(g)
        pos_mask, neg_mask = magneto_mask_from_disk(g, disk_mask, disk_mask_safe)

        # pos
        labels_p, points_p = cluster_dbscan(pos_mask)
        all_props_p = props_from_cluster(g, labels_p, points_p, pos_mask)
        all_props_p = add_helio_to_prop(obstime, a_disk, r_disk, c_disk, all_props_p)
        json_txt = os.path.join(OUT_BASE, "results_pos.txt")
        with open(json_txt, "a", encoding="utf-8") as f:
            for p in all_props_p:
                f.write(
                    f"{base_name}, G{p['label']}, "
                    f"deg(lat/lon)=({p['stonyhurst']['lat']:.2f}, {p['stonyhurst']['lon']:.2f}), "
                    f"deg(carr)={p['carrington']:.2f}, "
                    f"mu={p['mu']:.3f}, "
                    f"area_px={p['area_px']}, area_MSH={p['area_MSH']}, "
                    f"mean_bright={p['mean_bright']}, sum_bright={p['sum_bright']}\n")
        save_overlays(img, base_name, pos_mask, all_props_p, pos_neg='pos')

        # neg
        labels_n, points_n = cluster_dbscan(neg_mask)
        all_props_n = props_from_cluster(g, labels_n, points_n, neg_mask)
        all_props_n = add_helio_to_prop(obstime, a_disk, r_disk, c_disk, all_props_n)
        json_txt = os.path.join(OUT_BASE, "results_neg.txt")
        with open(json_txt, "a", encoding="utf-8") as f:
            for n in all_props_n:
                f.write(
                    f"{base_name}, G{n['label']}, "
                    f"deg(lat/lon)=({n['stonyhurst']['lat']:.2f}, {n['stonyhurst']['lon']:.2f}), "
                    f"deg(carr)={n['carrington']:.2f}, "
                    f"mu={n['mu']:.3f}, "
                    f"area_px={n['area_px']}, area_MSH={n['area_MSH']}, "
                    f"mean_bright={n['mean_bright']}, sum_bright={n['sum_bright']}\n")
        save_overlays(img, base_name, neg_mask, all_props_n, pos_neg='neg')

        # pair
        pairs = match_bipoles(all_props_p, all_props_n)
        pairs_out = os.path.join(OUT_BASE, "results_bipole.txt")
        with open(pairs_out, "a", encoding="utf-8") as fb:
            for (pp, nn, gcm, tilt) in pairs:
                fb.write(
                    f"{base_name}, P{pp['label']}-N{nn['label']}, "
                    f"tilt_deg={tilt:.2f}, gcm={gcm:.1f}\n")
        save_bipole_overlay(img, base_name, pairs)

    print(f"✅ 완료: {len(paths)}장 처리됨")


if __name__ == "__main__":
    #save_masks(IN)
    process_folder(IN)