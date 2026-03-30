# -*- coding: utf-8 -*-
"""
mcp.server.py

기능
- 전처리/매핑까지 반영된 CSV(mct_sample_with_persona_3_mapped_final.csv)를 로드
- 사용자 질의에서 브랜드명을 부분일치로 인식
- 해당 브랜드의 '슈머유형'과 'A_STAGE'에 따라 플랫폼 추천을 반환
- 배달매출비율이 높으면(>=50%) 배달 채널 추가

엔드포인트(툴)
- find_brands(query: str) -> 부분일치 후보 리스트
- recommend_channels(brand_query: str, prefer_stage: Optional[str]) -> 추천안(JSON)

사용 예
- recommend_channels("성우**")
- recommend_channels("스타벅", prefer_stage="A4")  # A_STAGE 우선 적용(없으면 기존 A_STAGE 사용)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from fastmcp.server import FastMCP, Context

# =========================
# 설정
# =========================
# 데이터 파일 경로를 스크립트 위치 기준으로 안전하게 계산
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "final_data.csv"  

# 브랜드 후보 컬럼 우선순위(존재하는 첫 컬럼 사용)
BRAND_COL_CANDIDATES = [
    "브랜드명",
    "가맹점명",
    "상호명",
    "ENCODED_MCT",
    "BRAND",
    "brand",
    "상호",
]

# 배달 비율 컬럼 후보
DELIVERY_RATIO_COLS = [
    "배달매출금액 비율",
    "배달매출 비율",
    "배달 비율",
]

# 슈머유형 & A-Stage 기반 채널 가이드라인 (LLM이 참고할 기본 방향성)
# 주의: 이는 "권장 사항"이며, Agent는 상황에 맞게 유연하게 조정할 수 있습니다.
RECO_TABLE: Dict[str, Dict[str, str]] = {
    "모스트슈머": {
        "A3": "인스타그램 릴스, 틱톡",
        "A4": "유튜브 쇼츠, 인스타그램 게시물",
        "A5": "인스타그램 게시물, 지역 카페",
    },
    "유틸슈머": {
        "A3": "네이버 블로그, 유튜브 영상",
        "A4": "네이버 블로그, 지역 카페",
        "A5": "지역 카페, 당근마켓",
    },
    "비지슈머": {
        "A3": "유튜브 쇼츠, 네이버 블로그",
        "A4": "유튜브 영상, 지역 카페",
        "A5": "네이버 블로그, 지역 카페",
    },
    "무소슈머": {
        "A3": "네이버 블로그, 지역 카페",
        "A4": "당근마켓, 네이버 블로그",
        "A5": "지역 카페, 당근마켓",
    },
}
BASE_CHANNEL = "네이버/카카오/구글맵, 리뷰노트"
DELIVERY_EXTRA = "배달의민족/쿠팡이츠"

# 슈머유형별 상세 특성 (LLM이 창의적으로 활용할 컨텍스트)
PERSONA_DETAILS = {
    "모스트슈머": {
        "demographics": "평균 연령 39.9세, 젊은 층",
        "media_behavior": "유튜브·인스타그램·OTT·블로그 모두 활발 이용, SNS·커뮤니티 참여도 높음",
        "key_traits": "트렌드 민감, 새로움 추구, 시각적 콘텐츠 선호, 빠른 정보 확산력",
        "marketing_context": "숏폼 비디오와 바이럴 콘텐츠에 반응 좋음, 인플루언서 마케팅 효과적"
    },
    "유틸슈머": {
        "demographics": "평균 연령 44.2세, 여성 비중 높음(56%)",
        "media_behavior": "포털(네이버·다음) 검색과 뉴스 중심, 메신저·콘텐츠 소비 활발, SNS·OTT는 낮음",
        "key_traits": "실용성·합리성 중시, 정보 기반 의사결정, 상세한 정보 선호",
        "marketing_context": "상세 리뷰와 정보성 콘텐츠 중요, 검색 최적화 필수, 입소문 마케팅 효과적"
    },
    "비지슈머": {
        "demographics": "평균 연령 42.4세, 남성 비중 높음(44%)",
        "media_behavior": "SNS(페이스북·인스타)·OTT 높음, 스트리밍·온라인쇼핑 자주, 뉴스·검색 낮음",
        "key_traits": "편의성·시간효율 중시, 빠른 의사결정, 모바일 중심",
        "marketing_context": "짧고 임팩트 있는 콘텐츠, 간편한 구매 프로세스, 타겟 광고 효과적"
    },
    "무소슈머": {
        "demographics": "평균 연령 46.4세, 가장 높은 연령대",
        "media_behavior": "전반적 이용률 낮음(SNS·OTT·게임·쇼핑 소극적), 정보검색·뉴스 위주",
        "key_traits": "보수적·신중한 소비, 검증된 정보 선호, 오프라인 선호",
        "marketing_context": "지역 기반 오프라인 마케팅, 신뢰 구축 중요, 입소문과 추천 효과적"
    }
}

# =========================
# 유틸
# =========================
def _to_percent100(x: Any) -> Optional[float]:
    """
    '57%', '57', '0.57' 등 -> 57.0
    변환 불가/결측 -> None
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().replace(",", "")
    if s == "" or s == "-999999.9":
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        v = float(s)
    except Exception:
        return None
    if 0.0 <= v <= 1.0:
        v *= 100.0
    return v


def _normalize(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return re.sub(r"\s+", "", str(text)).lower()


def _choose_brand_column(df: pd.DataFrame) -> Optional[str]:
    for c in BRAND_COL_CANDIDATES:
        if c in df.columns:
            return c
    # 브랜딩 관련 컬럼 추정(한글/영문 '명' 포함)
    for c in df.columns:
        if any(key in c for key in ["브랜드", "가맹점", "상호", "brand", "name"]):
            return c
    return None


def _choose_delivery_col(df: pd.DataFrame) -> Optional[str]:
    for c in DELIVERY_RATIO_COLS:
        if c in df.columns:
            return c
    # 추정
    for c in df.columns:
        if "배달" in c and "비율" in c:
            return c
    return None


def _stage_key(stage: str) -> str:
    """A_STAGE 값을 A3/A4/A5 키로 정규화"""
    s = (stage or "").upper()
    if s.startswith("A3"):
        return "A3"
    if s.startswith("A4"):
        return "A4"
    if s.startswith("A5"):
        return "A5"
    return ""


def _split_channels(s: str) -> List[str]:
    # "네이버 블로그, 유튜브 영상" -> ["네이버 블로그","유튜브 영상"]
    return [t.strip() for t in re.split(r"[,/]", s) if t.strip()]


# =========================
# 데이터 로드
# =========================
DF: Optional[pd.DataFrame] = None
BRAND_COL: Optional[str] = None
DELIV_COL: Optional[str] = None

def _load_df() -> pd.DataFrame:
    global DF, BRAND_COL, DELIV_COL
    DF = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    BRAND_COL = _choose_brand_column(DF)
    DELIV_COL = _choose_delivery_col(DF)
    return DF

# 초기 로드
_load_df()

# =========================
# MCP 서버
# =========================
mcp = FastMCP(
    "MerchantSearchServer",
    instructions="""
    브랜드(가맹점)를 찾아 '슈머유형'과 'A_STAGE'에 따라 마케팅 채널을 추천합니다.
    - find_brands: 부분일치 검색으로 브랜드 후보를 보여줍니다.
    - get_merchant: 가맹점(브랜드)명을 정확히 일치시켜 상세 레코드를 반환합니다.
    - recommend_channels: 브랜드명을 입력하면 추천 채널을 JSON으로 돌려줍니다.
    """
)

# -------------------------
# 통합검색 도구 
# -------------------------
@mcp.tool
def search_merchants(query: str) -> Dict[str, Any]:
    """
    가맹점명(부분/마스킹 일치) 또는 가맹점구분번호(완전 일치)로 가맹점을 검색합니다.
    검색 결과가 여러 개일 경우, 선택 가능한 목록을 반환합니다.
    
    매개변수:
      - query: 검색어 (예: "성우**", "16184E93D9")
    """
    if DF is None: _load_df()

    normalized_query = _normalize(query).replace("*", "")
    if not normalized_query:
        return {"count": 0, "merchants": [], "reason": "빈 검색어"}

    # 가맹점구분번호로 검색 시도 (숫자와 알파벳으로만 구성된 긴 문자열)
    if re.fullmatch(r'[a-z0-9]{10,}', normalized_query):
        mask = DF['가맹점구분번호'].astype(str).str.lower() == normalized_query
        matched_df = DF[mask]
    # 가맹점명으로 검색
    else:
        ser = DF[BRAND_COL].astype(str)
        mask = ser.map(lambda x: normalized_query in _normalize(x))
        matched_df = DF[mask]

    count = len(matched_df)
    if count == 0:
        return {"count": 0, "merchants": [], "reason": "검색 결과 없음"}

    # 사용자 선택에 필요한 최소한의 정보만 추출하여 반환
    merchants = matched_df[[
        '가맹점구분번호', BRAND_COL, '가맹점주소'
    ]].rename(columns={BRAND_COL: '가맹점명'}).to_dict(orient='records')[:10] # 최대 10개만

    return {
        "count": count,
        "merchants": merchants
    }

# -------------------------
# 검색: 부분일치 후보
# -------------------------
@mcp.tool
def find_brands(query: str) -> Dict[str, Any]:
    """
    부분일치로 브랜드 후보를 찾아 상위 20개 반환
    """
    if DF is None or BRAND_COL is None:
        _load_df()

    q = _normalize(query).replace("*", "")  # '성우**' 같은 마스킹 허용
    if not q:
        return {"ok": False, "reason": "빈 검색어"}

    ser = DF[BRAND_COL].astype(str)
    mask = ser.map(lambda x: _normalize(x).__contains__(q))
    hits = ser[mask].dropna().unique().tolist()[:20]
    return {
        "ok": True,
        "brand_column": BRAND_COL,
        "count": len(hits),
        "candidates": hits,
    }

# -------------------------
# 조회: 정확 일치로 레코드 가져오기
# -------------------------
@mcp.tool
def get_merchant(merchant_name: str) -> Dict[str, Any]:
    """
    브랜드/가맹점명을 정확히 일치시켜 해당 레코드들을 반환합니다.
    - 공백 제거/소문자화/마스킹(*) 제거 후 비교
    - 결과는 최대 50건 제한
    """
    if DF is None or BRAND_COL is None:
        _load_df()

    key = _normalize(merchant_name).replace("*", "")
    if not key:
        return {"ok": False, "reason": "빈 입력"}

    ser = DF[BRAND_COL].astype(str)
    norm = ser.map(lambda x: _normalize(x).replace("*", ""))
    mask = norm.eq(key)
    matched = DF[mask]
    records = matched.to_dict(orient="records")[:50]
    return {
        "ok": bool(records),
        "brand_column": BRAND_COL,
        "count": len(records),
        "records": records,
    }

# -------------------------
# 추천: 채널
# -------------------------
@mcp.tool
def recommend_channels(brand_query: str, prefer_stage: Optional[str] = None) -> Dict[str, Any]:
    """
    brand_query: 부분일치 문자열(예: '성우', '할매순대', '스타벅')
    prefer_stage: 'A3'|'A4'|'A5' 또는 'A3_Acquisition' 등(선택)
    """
    if DF is None or BRAND_COL is None:
        _load_df()

    q = _normalize(brand_query).replace("*", "")
    if not q:
        return {"ok": False, "reason": "빈 브랜드 질의"}

    # 매칭 행 추출: 정확 일치(정규화 & '*' 제거) 우선 -> 없으면 부분 일치
    ser = DF[BRAND_COL].astype(str)
    norm_ser = ser.map(lambda x: _normalize(x).replace("*", ""))
    exact_idx = norm_ser[norm_ser.eq(q)].index.tolist()
    match_mode = "exact"
    if exact_idx:
        i = exact_idx[0]
    else:
        part_idx = [ix for ix, name in ser.items() if q in _normalize(name)]
        if not part_idx:
            return {"ok": False, "reason": f"해당 브랜드를 찾을 수 없음: {brand_query}"}
        i = part_idx[0]
        match_mode = "partial"
    row = DF.loc[i]

    # 필수 값
    cluster = str(row.get("슈머유형", "") or "")
    a_stage_raw = str(row.get("A_STAGE", "") or "")
    stage_key = _stage_key(prefer_stage or a_stage_raw)  # 우선 prefer_stage

    # 추천 표 조회
    reco_by_stage = RECO_TABLE.get(cluster, {})
    stage_channels = reco_by_stage.get(stage_key, "")
    base_channels = BASE_CHANNEL
    delivery_channels = DELIVERY_EXTRA

    # 배달 비율 판단
    delivery_ratio = None
    if DELIV_COL and DELIV_COL in DF.columns:
        delivery_ratio = _to_percent100(row.get(DELIV_COL))
    include_delivery = delivery_ratio is not None and delivery_ratio >= 50.0

    # 페르소나 상세 정보 가져오기 (LLM이 창의적으로 활용할 컨텍스트)
    persona_detail = PERSONA_DETAILS.get(cluster, {
        "demographics": f"{cluster} 고객",
        "media_behavior": "정보 없음",
        "key_traits": "정보 없음",
        "marketing_context": "정보 없음"
    })
    
    # 응답 구성
    primary = _split_channels(stage_channels) if stage_channels else []
    base = _split_channels(base_channels)
    extra = _split_channels(delivery_channels) if include_delivery else []

    return {
        "ok": True,
        "brand_column": BRAND_COL,
        "brand": row.get(BRAND_COL),
        "match_mode": match_mode,
        "cluster_type": cluster,           # 슈머유형
        "persona_details": persona_detail, # 페르소나 상세 특성 (LLM이 창의적으로 활용)
        "a_stage": a_stage_raw,            # 원본 A_STAGE
        "stage_used": stage_key or None,   # 실제 추천에 사용한 스테이지 키(A3/A4/A5)
        "delivery_ratio_col": DELIV_COL,
        "delivery_ratio": delivery_ratio,
        "include_delivery_channels": include_delivery,
        "channel_suggestions": {
            "primary_by_stage": primary,   # 기본 추천 채널 (Agent가 조정 가능)
            "base_channels": base,         # 모든 가맹점 공통 기본 채널
            "delivery_additional": extra,  # 배달 비중 높을 시 추가 고려
        },
        "note_to_agent": "제시된 채널은 기본 가이드라인입니다. 페르소나 특성과 마케팅 상황을 종합적으로 고려하여 우선순위를 조정하거나 추가 채널을 제안할 수 있습니다."
    }

# -------------------------
# Q2
# -------------------------
@mcp.tool
def analyze_low_revisit_store(merchant_id: str) -> Dict[str, Any]:
    """
    재방문율이 낮은 특정 가맹점(merchant_id)의 7P 마케팅 믹스 지표를 종합적으로 분석하여 반환합니다. 
    
    매개변수:
      - merchant_id: 분석할 가맹점의 ID (가맹점구분번호)
    
    반환값:
      - 7P 분석 지표가 담긴 딕셔너리
    """
    if DF is None: _load_df()
    
    # 가맹점 구분번호는 문자열 타입으로 비교해야 정확합니다.
    store_data = DF[DF['가맹점구분번호'].astype(str) == str(merchant_id)]
    
    if len(store_data) == 0:
        return {"found": False, "message": f"'{merchant_id}' 가맹점을 찾을 수 없습니다."}
    
    # 동일 가맹점의 여러 월 데이터가 있을 경우, 최신 월을 기준으로 분석합니다.
    result = store_data.sort_values(by='기준년월', ascending=False).iloc[0].to_dict()
    
    # 에이전트가 분석하기 쉽도록 7P 기준으로 데이터를 구조화하여 반환합니다.
    # analyze_low_revisit_store 함수 내 report 생성 부분 수정
    report = {
        "found": True,
        "merchant_name": result.get("가맹점명"),
        "product": {
            "revisit_rank": result.get("PCT_REVISIT"), "revisit_cat": result.get("REVISIT_CAT"),
            "rtf_rank": result.get("PCT_RTF"), "rtf_cat": result.get("RTF_CAT"),
            "sales_rank": result.get("PCT_SALES"), "sales_cat": result.get("SALES_CAT"),
            "customer_type": result.get("CUSTOMER_TYPE") 
        },
        "price": {
            "price_rank": result.get("PCT_PRICE"), "price_cat": result.get("PRICE_CAT"),
            # 신규 Price KPI 추가
            "similar_price_ratio_rank": result.get("PCT_SIMILAR_PRICE"), 
            "similar_price_ratio_cat": result.get("SIMILAR_PRICE_CAT") 
        },
        "place": { # Place는 기존 TENURE만 유지
            "tenure_rank": result.get("PCT_TENURE"), "tenure_cat": result.get("TENURE_CAT")
        },
        "process": {
            "process_score_rank": result.get("PCT_PROCESS"), "process_cat": result.get("PROCESS_CAT")
        }
    }
    return report

# ---------------------------------
# 특화 질문 1: 핵심 고객 분석
# ---------------------------------
@mcp.tool
def analyze_main_customer_segment(merchant_id: str) -> Dict[str, Any]:
    """
    특정 가맹점(merchant_id)의 핵심 고객 그룹(상위 1~2개 연령/성별)과 
    주요 방문 유형(거주/직장/유동)을 분석하여 반환합니다.

    매개변수:
      - merchant_id: 분석할 가맹점의 ID (가맹점구분번호)

    반환값:
      - 핵심 고객 세그먼트 및 방문 유형 정보가 담긴 딕셔너리
    """
    if DF is None: _load_df() # 데이터 로드 확인

    # 가맹점 구분번호는 문자열로 비교
    store_data = DF[DF['가맹점구분번호'].astype(str) == str(merchant_id)]

    if len(store_data) == 0:
        return {"found": False, "message": f"'{merchant_id}' 가맹점을 찾을 수 없습니다."}

    # 최신 월 데이터 기준
    result = store_data.sort_values(by='기준년월', ascending=False).iloc[0].to_dict()

    # --- 상위 연령/성별 세그먼트 식별 ---
    age_gender_cols = [
        "남성 20대이하 고객 비중", "남성 30대 고객 비중", "남성 40대 고객 비중", 
        "남성 50대 고객 비중", "남성 60대이상 고객 비중",
        "여성 20대이하 고객 비중", "여성 30대 고객 비중", "여성 40대 고객 비중",
        "여성 50대 고객 비중", "여성 60대이상 고객 비중",
    ]

    segment_ratios = {}
    for col in age_gender_cols:
        ratio = result.get(col)
        # 결측치나 유효하지 않은 값은 0으로 처리하여 비교
        if pd.notna(ratio) and isinstance(ratio, (int, float)): 
            segment_ratios[col.replace(' 고객 비중','')] = float(ratio) # 깔끔한 이름과 값 저장
        else:
            segment_ratios[col.replace(' 고객 비중','')] = 0.0

    # 비율 기준으로 내림차순 정렬 후 상위 1~2개 추출
    sorted_segments = sorted(segment_ratios.items(), key=lambda item: item[1], reverse=True)

    top_segments = []
    if sorted_segments:
        top_segments.append(sorted_segments[0][0]) # 1위는 항상 추가
        # 2위가 존재하고 비율이 0보다 크면 추가
        if len(sorted_segments) > 1 and sorted_segments[1][1] > 0: 
            top_segments.append(sorted_segments[1][0])

    # --- 주요 방문 유형 식별 ---
    visit_type_cols = {
        "거주 이용 고객 비율": "거주 고객",
        "직장 이용 고객 비율": "직장 고객",
        "유동인구 이용 고객 비율": "유동인구 고객",
    }

    visit_ratios = {}
    for col, name in visit_type_cols.items():
        ratio = result.get(col)
        if pd.notna(ratio) and isinstance(ratio, (int, float)):
            visit_ratios[name] = float(ratio)
        else:
            visit_ratios[name] = 0.0

    # 가장 높은 비율의 방문 유형 찾기
    main_visit_type = max(visit_ratios, key=visit_ratios.get) if visit_ratios and sum(visit_ratios.values()) > 0 else "정보 없음"

    # 에이전트에게 전달할 보고서 구성
    report = {
        "found": True,
        # BRAND_COL이 None일 수 있으므로 '가맹점명'도 확인
        "merchant_name": result.get(BRAND_COL) if BRAND_COL else result.get("가맹점명", "이름 정보 없음"), 
        "top_segments": top_segments,         # 상위 1~2개 세그먼트 이름 리스트
        "main_visit_type": main_visit_type    # 가장 비중 높은 방문 유형 이름
    }
    return report

# -------------------------
# 특화 질문 2: 경쟁 우위 진단
# -------------------------
@mcp.tool
def analyze_competitive_positioning(merchant_id: str) -> Dict[str, Any]:
    """
    특정 가맹점(merchant_id)의 매출 및 가격 경쟁력(백분위 순위) 데이터를 조회합니다.
    
    매개변수:
      - merchant_id: 분석할 가맹점의 ID (가맹점구분번호)
    
    반환값:
      - 매출 및 가격 순위 데이터가 담긴 딕셔너리
    """
    if DF is None: _load_df() # 데이터 로드 확인
    
    # 가맹점 구분번호는 문자열로 비교
    store_data = DF[DF['가맹점구분번호'].astype(str) == str(merchant_id)]
    
    if len(store_data) == 0:
        return {"found": False, "message": f"'{merchant_id}' 가맹점을 찾을 수 없습니다."}
    
    # 최신 월 데이터 기준
    result = store_data.sort_values(by='기준년월', ascending=False).iloc[0].to_dict()
    
    # 에이전트가 사용할 핵심 경쟁 지표만 추출하여 반환
    report = {
        "found": True,
        "merchant_name": result.get("가맹점명"),
        "sales_rank_percentile": result.get("PCT_SALES"), 
        "sales_cat": result.get("SALES_CAT"), # 추가
        "price_rank_percentile": result.get("PCT_PRICE"),
        "price_cat": result.get("PRICE_CAT") # 추가
    }
    return report

# -------------------------
# 헬스체크 / 리로드
# -------------------------
@mcp.tool
def ping() -> str:
    return "pong"


@mcp.tool
def reload_data() -> Dict[str, Any]:
    df = _load_df()
    return {
        "ok": True,
        "rows": int(len(df)),
        "brand_col": BRAND_COL,
        "delivery_col": DELIV_COL,
        "columns": df.columns.tolist(),
    }


# ==============================
# Q3: 현재 가장 큰 문제점 자동 진단
# ==============================

def _to_pct(v):
    if pd.isna(v):
        return None
    try:
        v = float(v)
    except Exception:
        return None
    if v <= 1.0:
        return v * 100.0
    return v

_Q3_CONTROLLABILITY = {
    "Price": 50.0,
    "Place": 35.0,
    "Promotion": 100.0,
    "Process": 75.0,
}

_Q3_DIM_COLS = {
    "Price": ["PCT_SALES", "PCT_PRICE"],
    "Place": ["PCT_TENURE", "PCT_CLOSURE", "PCT_ACCESS"],
    "Promotion": ["PCT_REVISIT", "PCT_RTF", "CRI_PCT", "PCT_A3A4", "PCT_A4A5"],
    "Process": ["PCT_PROCESS", "PROCESS_SCORE_PCT"],
}

def _severity_from_series(s: pd.Series, controllability: float) -> Dict[str, Any]:
    vals = s.dropna().tolist()
    if not vals:
        return {"severity": None, "impact": None, "duration": None, "count": 0}
    impact_vals = [max(0.0, 100.0 - _to_pct(v)) for v in vals]
    # 최근 6개월 위험 비율은 상위에서 별도로 계산하고 이 함수에서는 단일 시점만 계산
    impact = sum(impact_vals) / len(impact_vals)
    # duration은 호출부에서 전달
    return {"impact": impact, "controllability": controllability}

def _calc_duration(last6: pd.Series) -> float:
    # 위험구간 PR<30 비율
    if last6.empty:
        return 0.0
    cnt = last6.apply(lambda v: (_to_pct(v) or 0) < 30.0).sum()
    return (cnt / len(last6)) * 100.0

@mcp.tool(name="analyze_q3", description="Q3: {요식업종 가맹점}의 현재 가장 큰 문제점을 진단하고 근거와 함께 반환합니다. 입력: merchant_id(str).")
def analyze_q3(merchant_id: str) -> Dict[str, Any]:
    """
    입력: merchant_id (e.g., 가맹점구분번호)
    출력: {
      ok: bool,
      merchant_id, merchant_name, baseline_month,
      step1: { Price: [{col, pr}], ... },
      step2: { severity_by_P: {...}, current_key_issue: str },
      step3: { recommended_strategies: [...], evidence_columns: [...] }
    }
    """
    if DF is None:
        _load_df()
    df = DF

    # 식별자/이름/월 컬럼 후보
    id_cols = ["가맹점구분번호"]
    name_cols = ["가맹점명"]
    month_cols = ["기준년월"]

    id_col = next((c for c in id_cols if c in df.columns), None)
    if id_col is None:
        return {"ok": False, "error": "식별자 컬럼이 없습니다. (가맹점구분번호/MERCHANT_ID 등)"}

    sub = df[df[id_col].astype(str) == str(merchant_id)].copy()
    if sub.empty:
        return {"ok": False, "error": f"merchant_id={merchant_id} 데이터가 없습니다."}

    # 최신월 기준
    mcol = next((c for c in month_cols if c in sub.columns), None)
    if mcol:
        sub[mcol] = sub[mcol].astype(str)
        sub = sub.sort_values(mcol)
        recent = sub.iloc[-1]
        # 최근 6개월 슬라이스
        last6 = sub.iloc[-6:]
        baseline_month = recent[mcol]
    else:
        recent = sub.iloc[-1]
        last6 = sub.iloc[-6:]
        baseline_month = None

    name_col = next((c for c in name_cols if c in df.columns), None)
    merchant_name = str(recent[name_col]) if name_col else None

    # STEP1: P별 현재 PR 나열
    step1 = {}
    for dim, cols in _Q3_DIM_COLS.items():
        present = []
        for c in cols:
            if c in sub.columns:
                present.append({"col": c, "pr": _to_pct(recent[c])})
        if present:
            step1[dim] = present

    if not step1:
        return {"ok": False, "error": "Q3에 사용할 지표 컬럼이 없습니다. (PCT_*, *_PCT 등)"}

    # STEP2: 심각도 산출
    severity_by_P = {}
    details = {}
    for dim, cols in _Q3_DIM_COLS.items():
        vals = []
        evid = []
        for c in cols:
            if c in sub.columns:
                controllability = _Q3_CONTROLLABILITY.get(dim, 50.0)
                # duration: 최근 6개월 기준
                dur = _calc_duration(last6[c]) if c in last6.columns else 0.0
                impact = max(0.0, 100.0 - (_to_pct(recent[c]) or 0.0))
                sev = 0.55*impact + 0.25*dur + 0.20*controllability
                vals.append(sev)
                evid.append({"col": c, "impact": impact, "duration": dur, "controllability": controllability, "severity": sev})
        if vals:
            severity_by_P[dim] = sum(vals)/len(vals)
            details[dim] = evid

    if not severity_by_P:
        return {"ok": False, "error": "심각도 계산에 사용할 유효 지표가 없습니다."}

    current_key_issue = max(severity_by_P, key=severity_by_P.get)

    # STEP3: 전형 전략 매핑
    strategies = {
        "Price": [
            "가격·메뉴 구조 재정비: 인기 메뉴 소형화/세트화로 객단가 탄력 확보",
            "특정 요일·시간대 차등가로 한가한 시간 수요 이동",
            "원가 구조 투명화 후 원재료 대체/공급선 다변화"
        ],
        "Place": [
            "상권 브랜딩 협업(동네 축제/상점가 공동 프로모션)으로 외부 유입 보완",
            "근거리 타겟 쿠폰(반경 500m) 발행으로 생활권 점유율 확대",
            "접근성 약점 보완: 역세권/정류장 광고 또는 지도앱 스폰서"
        ],
        "Promotion": [
            "첫 방문 경험 설계: 첫 2회차 쿠폰·스탬프 제공으로 A3→A4 전환 촉진",
            "리뷰/멤버십·N회차 리워드로 A4→A5 충성화 가속",
            "메신저·카톡채널·지역 커뮤니티 리마인더로 재구매 주기 단축"
        ],
        "Process": [
            "피크타임 메뉴 단순화 및 선결제/QR주문으로 대기시간 단축",
            "CS 표준화(제공시간 SLA·재조리 원칙)와 직원 교차교육",
            "재고/발주 자동화로 품절·취소율 저감"
        ]
    }
    recommended = strategies.get(current_key_issue, [])

    return {
        "ok": True,
        "merchant_id": str(merchant_id),
        "merchant_name": merchant_name,
        "baseline_month": baseline_month,
        "step1": step1,
        "step2": {
            "severity_by_P": severity_by_P,
            "details": details,
            "current_key_issue": current_key_issue
        },
        "step3": {
            "recommended_strategies": recommended,
            "evidence_columns": [e["col"] for e in sum(details.values(), [])]
        }
    }
if __name__ == "__main__":
    mcp.run()
