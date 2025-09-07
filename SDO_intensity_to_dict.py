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



# ============== 설정 ==============
IN = r'D:\SDO\image\hmiif'                 # 이미지 폴더
OUT_BASE = r'D:\SDO\image\hmiif\mask_overlays'   # 출력 저장 루트
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
# ============== (2) 태양 흑점 마스킹 ==============
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


def sunspot_mask_from_disk(g, disk_mask, disk_mask_safe): 
    """중앙값 기반 로버스트 통계 이진화로 전처리 후, 임계값으로 흑점 암부/반암부 구별"""
    # 전처리 
    disk_val = as_float(g[disk_mask > 0])     # 태양 컨투어 내부만 
    med = np.median(disk_val)                 # 평균 대신 중앙값
    MAD = np.median(np.abs(disk_val - med))   # 표준편차 대신 중앙값 절대 편차
    sigma = 1.4826 * MAD + 1e-6               # 정규분포일 때 표준편차의 로버스트 추정치
    
    # 마스크 생성
    Tu = med - 11*sigma   # umbra 임계값 (이거보단 어두워야 함)
    umbra_mask = zero_img(g)
    umbra_mask[(disk_mask>0) & (g < Tu)] = 255
    umbra_mask = cv.bitwise_and(umbra_mask, disk_mask_safe)

    Tp = med - 6*sigma   # penumbra 임계값 (이거보단 어두워야 함)
    penum_mask = zero_img(g)
    penum_mask[(disk_mask>0) & (g >= Tu) & (g < Tp)] = 255
    penum_mask = cv.bitwise_and(penum_mask, disk_mask_safe)

    return umbra_mask, penum_mask


def save_masks(in_dir):
    """mask 이미지 저장"""
    OUT_DIRS = {"disk":  os.path.join(OUT_BASE, "disk_mask"),
                "umbra": os.path.join(OUT_BASE, "umbra_mask"),
                "penum": os.path.join(OUT_BASE, "penum_mask")}
    for d in OUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

    paths = glob.glob(os.path.join(in_dir, "*.jpg"))   # 폴더안 모든 jpg 경로 불러오기
    for p in paths:
        img = cv.imread(p, cv.IMREAD_COLOR)
        g = gray(img)
        base_name = os.path.splitext(os.path.basename(p))[0]   # 이미지 이름만 

        _, _, _, disk_mask, disk_mask_safe = sundisk_mask_from_img(g)
        umbra_mask, penum_mask = sunspot_mask_from_disk(g, disk_mask, disk_mask_safe)

        cv.imwrite(os.path.join(OUT_DIRS["disk"],  base_name + "_disk.jpg"),  disk_mask)
        cv.imwrite(os.path.join(OUT_DIRS["umbra"], base_name + "_umbra.jpg"), umbra_mask)
        cv.imwrite(os.path.join(OUT_DIRS["penum"], base_name + "_penum.jpg"), penum_mask)
    print(f"✅ 완료: {len(paths)}장 처리됨")



# ============================================
# ============== (3) 흑점 군집화 ==============
# ============================================
def cluster_dbscan(umbra_mask, penum_mask):
    """umbra&penum 합친 마스크를 기반으로 포인트를 추출해서 DBSCAN 군집화"""
    spot_mask = cv.bitwise_or(umbra_mask, penum_mask)           # 합치기
    spot_mask = cv.dilate(spot_mask, cv.getStructuringElement(  # 살짝 팽창 (반암부 연결)
                              cv.MORPH_ELLIPSE,(5,5)), iterations=1)
    ys, xs = np.where(spot_mask > 0)
    points = np.stack([xs, ys], axis=1) if len(xs) else np.empty((0,2), dtype=int)
    if len(points)==0: return np.array([]), points   # 군집 없을 때
    
    db = DBSCAN(eps=10, min_samples=3)
    labels = db.fit_predict(points)
    return labels, points


def tilt_from_mask(cl_mask):
    """군집 마스크의 주성분(가장 긴 방향)으로부터 tilt 각도를 계산"""
    ys, xs = np.where(cl_mask > 0)
    if len(xs) < 2:
        return float('nan')
    X = np.stack([xs, ys], axis=1).astype(float)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD = PCA
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    vx, vy = Vt[0]   # 제1고유벡터
    # OpenCV 이미지 좌표계(y down) 보정 위해 -vy 사용
    angle = math.degrees(math.atan2(-vy, vx))
    # [-90, +90] 범위로 정규화
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return float(angle)


def props_from_cluster(g, labels, points, umbra_mask, penum_mask):
    """군집별 기울기, 밝기, 면적, 무게중심 좌표를 계산"""
    all_props = []
    if len(points)==0: return all_props   # 군집 없을 때
    for k in sorted([l for l in np.unique(labels) if l!=-1]):
        idx = np.where(labels==k)[0]
        pts = points[idx]

        # 군집 마스크 만들기
        cl_mask = zero_img(umbra_mask)
        cl_mask[pts[:,1], pts[:,0]] = 255
        cl_mask = cv.morphologyEx(cl_mask, cv.MORPH_CLOSE, 
                                  cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)),1)

        # 기울기
        tilt_deg = tilt_from_mask(cl_mask)

        # 마스크 밝기
        ys_u, xs_u = np.where((cl_mask>0) & (umbra_mask>0))            # 암부 밝기 
        vals_u = g[ys_u, xs_u]
        mean_bright_u = float(np.mean(vals_u)) if len(vals_u)>0 else 0.0
        sum_bright_u  = float(np.sum(vals_u))  if len(vals_u)>0 else 0.0

        ys_p, xs_p = np.where((cl_mask>0) & (penum_mask>0))            # 반암부 밝기 
        vals_p = g[ys_p, xs_p]
        mean_bright_p = float(np.mean(vals_p)) if len(vals_p)>0 else 0.0
        sum_bright_p  = float(np.sum(vals_p))  if len(vals_p)>0 else 0.0
    
        quiet_mask = (cl_mask>0) & (umbra_mask==0) & (penum_mask==0)   # 로컬 배경 밝기
        vals_q = g[quiet_mask]
        mean_bright_q = float(np.mean(vals_q)) if len(vals_q)>0 else 0.0
        sum_bright_q  = float(np.sum(vals_q))  if len(vals_q)>0 else 0.0

        # 마스크 면적
        area_u = int(np.sum((cl_mask>0) & (umbra_mask>0)))
        area_p = int(np.sum((cl_mask>0) & (penum_mask>0)))
        area_t = area_u + area_p
        if area_t < 3: continue   # 면적<4인 군집은 스킵

        # 마스크 무게중심
        m = cv.moments(cl_mask, binaryImage=True)
        cx = m['m10']/m['m00']; cy = m['m01']/m['m00']

        # 마스크 바운딩박스
        x1, y1 = int(pts[:,0].min()), int(pts[:,1].min())
        x2, y2 = int(pts[:,0].max()), int(pts[:,1].max())
        bbox = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

        all_props.append({"label": int(k), 
                          "centroid_px": (float(cx), float(cy)), "bbox": bbox,
                          "area_px": {"umbra": area_u, "penumbra": area_p, "total": area_t},
                          "mean_bright": {"umbra": mean_bright_u, "penumbra": mean_bright_p, "quiet": mean_bright_q},
                          "sum_bright": {"umbra": sum_bright_u, "penumbra": sum_bright_p, "quiet": sum_bright_q},
                          "tilt_deg": tilt_deg,
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
        area_u_mu = p['area_px']['umbra']    / mu
        area_p_mu = p['area_px']['penumbra'] / mu
        area_t_mu = p['area_px']['total']    / mu

        # 관측되는 앞면 반구 전체를 백만으로 두는 단위 변환 (Millionths of Solar Hemisphere)
        MSH_U = float(area_u_mu / a_disk * 1e6/2)
        MSH_P = float(area_p_mu / a_disk * 1e6/2)
        MSH_T = float(area_t_mu / a_disk * 1e6/2)

        p["stonyhurst"] = {"lat": LAT, "lon": LON}
        p["carrington"] = LON_CA
        p["mu"] = mu
        p["area_MSH"] = {"umbra": MSH_U, "penumbra": MSH_P, "total": MSH_T}
    return all_props


def save_overlays(img, base_name, umbra_mask, penum_mask, all_props):
    """마스크 및 군집 바운딩박스 오버레이 이미지 저장"""
    OUT_DIR = os.path.join(OUT_BASE, "overlays")
    os.makedirs(OUT_DIR, exist_ok=True)
    ov_img = img.copy()

    # 흑점 마스크 오버레이
    color = np.array([0, 0, 255], dtype=np.uint8)
    ov_img[penum_mask > 0] = as_unit(as_float(0.6*ov_img[penum_mask > 0]) + 0.4*color)
    ov_img[umbra_mask > 0] = as_unit(as_float(0.3*ov_img[umbra_mask > 0]) + 0.7*color)

    for p in all_props:
        # mask에 딱 맞는 박스 그리기
        x, y, w, h = p["bbox"]
        cv.rectangle(ov_img, (x, y), (x + w, y + h), (0,255,0), 2)

        # 무게중심 표시 
        cx,cy = p["centroid_px"]
        cv.circle(ov_img, (int(round(cx)),int(round(cy))), 3, (255,255,255), -1)
        
        txt = (f"H:{p['stonyhurst']['lat']:.1f}, "
               f"C:{p['carrington']:.1f}, "
               f"A:{p['area_px']['total']}, ")
        cv.putText(ov_img, txt, (x, max(y-4, 10)), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)
    cv.imwrite(os.path.join(OUT_DIR, base_name + "_overlay.jpg"), ov_img)



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
        umbra_mask, penum_mask = sunspot_mask_from_disk(g, disk_mask, disk_mask_safe)

        labels, points = cluster_dbscan(umbra_mask, penum_mask)
        all_props = props_from_cluster(g, labels, points, umbra_mask, penum_mask)
        all_props = add_helio_to_prop(obstime, a_disk, r_disk, c_disk, all_props)

        # 저장
        json_txt = os.path.join(OUT_BASE, "results.txt")
        with open(json_txt, "a", encoding="utf-8") as f:
            for q in all_props:
                f.write(
                    f"{base_name}, G{q['label']}, "
                    f"deg(lat/lon)=({q['stonyhurst']['lat']:.2f}, {q['stonyhurst']['lon']:.2f}), "
                    f"deg(carr)={q['carrington']:.2f}, "
                    f"mu={q['mu']:.3f}, "
                    f"area_px={q['area_px']}, area_MSH={q['area_MSH']}, "
                    f"mean_bright={q['mean_bright']}, sum_bright={q['sum_bright']}, "
                    f"tilt_deg={q.get('tilt_deg','nan')}\n")
        save_overlays(img, base_name, umbra_mask, penum_mask, all_props)

    print(f"✅ 완료: {len(paths)}장 처리됨")


if __name__ == "__main__":
    save_masks(IN)
    process_folder(IN)