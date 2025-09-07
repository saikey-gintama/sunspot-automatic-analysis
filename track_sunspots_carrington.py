from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

SRS_DIR = Path(r"D:\SDO\image\hmib\mask_overlays")

import json
OUT_PJSON = Path(r"D:\SDO\image\hmib\mask_overlays\parents_201201n202410_2day.json")
OUT_GPJSON = Path(r"D:\SDO\image\hmib\mask_overlays\grandparents_201201n202410_2day.json")

# ========================================
# -------- 1) ID 모든 흑점 불러오기 -------- 
# ========================================
# all_rows (ID = YYYYMMDD_nmbr)
"""일별 관측된 모든 흑점 리스트"""

all_rows = {}
for path in sorted(SRS_DIR.glob("*.txt")):

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    date_str = path.stem[:8]
    date = datetime.strptime(date_str, "%Y%m%d").date()

    # "I."과 "IA." 사이 데이터만 추출 
    try:
        i_start = next(i for i, l in enumerate(lines) if l.startswith("I."))
        i_end   = next(i for i, l in enumerate(lines) if l.startswith("IA."))
    except StopIteration:
        print(f"⚠ {path.name} — 'I.' 또는 'IA.' 섹션 없음, 건너뜀")
        continue
    data_lines = lines[i_start+2 : i_end]  # i_start+1 → 헤더, 그 다음줄부터 데이터

    parsed_line = [line.split() for line in data_lines]
    for row in parsed_line:
        if len(row) >= 8:
            nmbr, helio_loca, carr_lo, area, z, horizon_ll, f_nn, mag_type = row[:8]
            if '*' in helio_loca: continue  # W0* 같은 행은 무시
            ID = f"{date_str}_{nmbr}"
            all_rows[ID] = {"nmbr": nmbr,
                            "helio_lat": float(helio_loca[:3].replace('N', '').replace('S', '-')), 
                            "helio_lo":  str(helio_loca[3:]),
                            "helio_loca": helio_loca, "carr_lo": carr_lo,
                            "area": area, "z": z,
                            "horizon_ll": horizon_ll, "f_nn": f_nn,
                            "mag_type": mag_type}         
#print(all_rows["20230707_3363"])    #‼️‼️‼️{'nmbr': '3363', 'helio_lat': -21.0, 'helio_lo': 'E68', ...}



def list_values(key):
    """부모ID 상관없이 해당 key에 속한 ✨값들을 전부 나열"""
    return [all_dict[key] for all_dict in all_rows.values()] 
#print(list_values("nmbr"))          # ['3341', '3345', '3354', ...]
#print(list_values("helio_loca"))    # ['S16W75', 'N06W78', 'N13W30', ...]

def find_contains(key, val):
    """key 값에 val(부분문자열 가능)이 포함된 항목의 ✨부모 ID 목록"""
    return [ID for ID, all_dict in all_rows.items() if str(val) in str(all_dict.get(key, ""))]
#print(find_contains("nmbr", "3363"))       # ['20230707_3363', '20230708_3363', '20230709_3363', ...]
#print(find_contains("helio_loca", "S"))    # ['20230701_3341', '20230701_3355', '20230701_3356', ...]
#print(find_contains("z", "C"))             # ['20230702_3356', '20230705_3360', '20230706_3360', ...]



# =============================================
# -------- 2) IDID 흑점넘버 기준으로 묶기 -------- 
# =============================================
# parents (IDID = startDate_endDate_nmbr)
"""같은 흑점 번호는 한 세트로 연결"""

by_nmbr = defaultdict(list)
for ID, _ in all_rows.items():
    date, nmbr = ID.split('_', 1)
    date = datetime.strptime(date, "%Y%m%d").date()
    by_nmbr[nmbr].append((date, ID))  # 같은 흑점번호를 하나의 리스트로 합치기 

parents = {}
for nmbr, IDs in by_nmbr.items():
    IDs.sort(key=lambda x: x[0])  # 날짜순 정렬 

    run_ID = [IDs[0][1]]  # 흑점번호묶음 하나 불러오기 (IDs)
    run_start = before = IDs[0][0]

    for after, ID in IDs[1:]:
        if (after - before) == timedelta(days=1):
            run_ID.append(ID)
        else:
            # 연속 끊김: 지금까지의 런s 마감
            IDID = f"{run_start:%Y%m%d}_{before:%Y%m%d}_{nmbr}"
            parents[IDID] = {ID: all_rows[ID] for ID in run_ID}

            # 새 런 시작
            run_ID = [ID]
            run_start = after
        before = after
    
    # 마지막 런 마감
    IDID = f"{run_start:%Y%m%d}_{before:%Y%m%d}_{nmbr}"
    parents[IDID] = {ID: all_rows[ID] for ID in run_ID}

#print(parents["20230707_20230719_3363"])    #‼️‼️‼️{'20230707_3363': {'nmbr':...'Alpha'}, '20230708_3363': {'nmbr':...'Alpha'}, ...}


# ===============================================
# -------- 3) IDIDID 회귀흑점 기준으로 묶기 -------- 
# ===============================================
# grandparents (IDIDID = firstDate(nmbr)_lastDate(nmbr)_수명)
""" 1. 같은 흑점번호(IDID) 속 마지막 관측에서 helio_lo가 ✨W75~W100이고 area가 ✨10 이상인 흑점들 중, 
    2. ✨13일~16일 뒤에 helio_lat이 ✨2도 이하로 비슷하고 carr_lo가 ✨10도 이하로 비슷하면서 helio_lo가 ✨E100~E50인 흑점이 있다면, 
    Δt ∈ [13, 16], |Δlat| ≤ 2°, circ_diff(Δcarr_lo) ≤ 10°
    3. 그 흑점의 흑점번호(IDID)를 묶어 회귀흑점번호IDIDID를 부여한다
    4. 일치하는 후보가 정확히 1개면 매칭 확정 및 재매칭 금지, 0개면 넘어감, 여러 개면 경고 후 사람 검토 
    5. 방금 만든 IDIDID의 마지막 멤버를 새로운 출발점으로 삼아 동일 규칙을 반복 (2회전 이상일 경우 고려)"""

info, parents_index = {}, defaultdict(list)
for IDID, IDs in parents.items():
    s,e,n = IDID.split('_',2)
    sd=datetime.strptime(s,"%Y%m%d").date()
    ed=datetime.strptime(e,"%Y%m%d").date()

    k0, k1 = f"{s}_{n}", f"{(sd+timedelta(1)):%Y%m%d}_{n}"  # start 다음날
    k2, k3 = f"{e}_{n}", f"{(ed-timedelta(1)):%Y%m%d}_{n}"  # end 전날

    if k1 not in IDs or k3 not in IDs: continue
    s0,s1,e0,e1 = IDs[k0], IDs[k1], IDs[k2], IDs[k3]
    info[IDID] = {
        "nmbr":n, "start":sd, "end":ed, "start_str":s, "end_str":e,
        "start_lo":  s0["helio_lo"], "end_lo":e0["helio_lo"], "end_area":int(e0["area"]),
        "start_lat": (float(s0["helio_lat"]) + float(s1["helio_lat"])) / 2,
        "start_lon": (float(s0["carr_lo"])   + float(s1["carr_lo"]))   / 2,
        "end_lat":   (float(e0["helio_lat"]) + float(e1["helio_lat"])) / 2,
        "end_lon":   (float(e0["carr_lo"])   + float(e1["carr_lo"]))   / 2}
    parents_index[sd].append(IDID)


grandparents, used = {}, set()
for IDID, meta in sorted(info.items(), key=lambda kv: kv[1]["end"]):  # 날짜순 정렬 
    if IDID in used: continue

    # 마지막 관측이 W75~W100이고 area>=10인가?
    EW, EW_lo = meta["end_lo"][0], int(meta["end_lo"][1:])
    if not ((EW == 'W' and 75 <= EW_lo <= 100) and (meta["end_area"] >= 10)):
        continue

    run_IDID = [IDID]
    used.add(IDID)

    while True:
        # IDID 마지막 관측 (사라진 날) 경계정보 불러오기 
        before  = info[run_IDID[-1]]["end"]  
        end_lat = info[run_IDID[-1]]["end_lat"]
        end_lon = info[run_IDID[-1]]["end_lon"]

        candidates = []
        # Δt=13~16 이후 시작하는 IDID가 있나? 
        for d in range(13, 17):  
            after = before + timedelta(days=d)
            for candi_IDID in parents_index.get(after, []):
                if candi_IDID in used: continue
                candi_info = info[candi_IDID]

                # 재등장 첫 관측이 E100~E50인가?
                EW, EW_lo = candi_info["start_lo"][0], int(candi_info["start_lo"][1:])
                if not (EW == 'E' and 50 <= EW_lo <= 100):
                    continue

                # |Δlat| ≤ 2°에 속하나?
                if abs(candi_info["start_lat"] - end_lat) > 2.0:
                    continue

                # circ_diff(Δcarr_lo) ≤ 10°에 속하나? 
                dlon = abs(candi_info["start_lon"] - end_lon) % 360.0
                if dlon > 180.0:
                    dlon = 360.0 - dlon
                if dlon <= 10.0:
                    candidates.append((candi_IDID, candi_info, d, dlon, abs(candi_info["start_lat"] - end_lat)))
        if not candidates: break

        # 다중 후보일 경우 경도차 작은 후보를 선택 
        if len(candidates) > 1: 
            print("\n[다중 후보] 사람 검토 필요")
            print(f"  기준 IDID  : {run_IDID[-1]}")
            print(candidates)
        
            nxt, nfo, d, dlon, dlat = min(candidates, key=lambda t: t[3])
            print(nxt, nfo, d, dlon, dlat)
        else:
            # 유일 후보
            nxt, nfo, d, dlon, dlat = candidates[0]

        run_IDID.append(nxt)
        used.add(nxt)

        # 후보 런의 마지막이 W75~W100이고 area>=10이면 하나 더 찾음
        EW2, EW_lo2 = nfo["end_lo"][0], int(nfo["end_lo"][1:])
        if not ((EW2 == 'W' and 75 <= EW_lo2 <= 100) and (nfo["end_area"] >= 10)):
            break

    # 회귀흑점이면 IDIDID 생성
    if len(run_IDID) >= 2:
        first = info[run_IDID[0]]   # IDIDID 첫 관측 (태어난 날)
        last  = info[run_IDID[-1]]  # IDIDID 마지막 관측 (죽은 날)
        total_days = (last["end"] - first["start"]).days + 1  # 수명 계산
        IDIDID = f"{first['start_str']}({first['nmbr']})_{last['end_str']}({last['nmbr']})_{total_days}"
        grandparents[IDIDID] = {k: parents[k] for k in run_IDID}

#print(grandparents['20230707(3363)_20230815(3394)_40'])    #‼️‼️‼️{'20230707_20230719_3363': {'20230707_3363': {'nmbr':}}, ... '20230804_20230815_3394': {'20230804_3394': {'nmbr':...}}}



def gp_overview():
    """전체 회귀흑점(IDIDID) 목록과 각 그룹의 IDID 목록을 출력"""
    print(f"🔗 회귀흑점(grandparents) 수: {len(grandparents)}")
    items = sorted(grandparents.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for i, (gid, runs) in enumerate(items, 1):
        members = list(runs.keys())  # 순서 유지
        print(f"{i:3d}. {gid}  (len={len(members)})")  # IDIDID 목록
        for j, IDID in enumerate(members, 1):
            print(f"     {j:2d}) {IDID}")              # IDID 목록

            recs = parents[IDID].values()
            lats  = [float(r["helio_lat"]) for r in recs]
            carrs = [float(r["carr_lo"])   for r in recs]           
            lat_avg  = (sum(lats) / len(lats)) if lats else float('nan')
            lat_rng  = (max(lats) - min(lats)) if lats else 0.0
            carr_avg = (sum(carrs) / len(carrs)) if carrs else float('nan')
            carr_rng = (max(carrs) - min(carrs)) if carrs else 0.0
            # ※ carr_lo가 0/360 경계 걸치면 범위가 커질 수 있음
            lo_key = "start_lo" if j == 1 else "end_lo"
            lo_val = info[IDID][lo_key]
            print(f"           | "
                  f"lat {lat_avg:+.1f} (Δ{lat_rng:.1f}) | "
                  f"carr {carr_avg:.0f} (Δ{carr_rng:.0f}) | "
                  f"{lo_key}={lo_val}")

gp_overview()



# =========================
# -------- 4) 저장 -------- 
# =========================
with open(OUT_PJSON, "w", encoding="utf-8") as f:
    json.dump(parents, f, ensure_ascii=False, indent=2)


with open(OUT_GPJSON, "w", encoding="utf-8") as f:
    json.dump(grandparents, f, ensure_ascii=False, indent=2)