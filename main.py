import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re
import google.generativeai as genai
from lunar_python import Solar, Lunar  # 만세력 라이브러리
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from saju_counseling_engine import (
    build_personalized_emergency_report,
    build_dynamic_gemini_prompt,
    DAY_MASTER_META
)

# -------------------------------------------------------------
# 0. 환경 변수(.env) 로드 및 보안 설정
# -------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일 또는 클라우드 대시보드를 확인하세요.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# 포트원(아임포트) 결제 연동 환경변수
PORTONE_API_KEY = os.getenv("PORTONE_API_KEY", "")
PORTONE_API_SECRET = os.getenv("PORTONE_API_SECRET", "")
PORTONE_STORE_ID = os.getenv("PORTONE_STORE_ID", "")

# -------------------------------------------------------------
# 1. 데이터베이스 세팅 (SQLite & SQLAlchemy)
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vercel / AWS Lambda 등 서버리스 환경(읽기 전용 파일시스템) 대응:
# VERCEL 환경변수가 있거나 BASE_DIR에 쓰기 권한이 없는 경우 /tmp에 DB 생성
if os.getenv("VERCEL") or not os.access(BASE_DIR, os.W_OK):
    DB_PATH = os.path.join("/tmp", "saju_database.db")
else:
    DB_PATH = os.path.join(BASE_DIR, "saju_database.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy SajuHistory 모델 정의
class SajuHistory(Base):
    __tablename__ = "saju_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    gender = Column(String(10), nullable=False)
    birth_date = Column(String(20), nullable=False)
    birth_time = Column(String(10), nullable=False)
    birth_type = Column(String(10), nullable=False)
    ai_interpretation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

# 테이블 자동 생성 (서버리스 환경에서 예외 발생 시 서비스 중단 방지)
try:
    Base.metadata.create_all(bind=engine)
except Exception as db_init_err:
    print(f"[WARNING] Database initialization skipped or failed: {db_init_err}")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# 관리자 인증 의존성
def verify_admin(x_api_key: Optional[str] = Header(None)):
    """ADMIN_API_KEY 환경변수가 설정된 경우에만 API 키 인증 요구"""
    if ADMIN_API_KEY and x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="관리자 인증이 필요합니다.")

# DB 세션 의존성 주입 (Dependency)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------------------
# 2. FastAPI 앱 초기화 및 CORS 미들웨어 설정
# -------------------------------------------------------------
app = FastAPI(title="Saju AI Backend API", version="1.0.0")

# 🌟 [배포 대비] CORS(교차 출처 리소스 공유) 미들웨어 설정
# ALLOWED_ORIGINS 환경변수로 허용 도메인 제어 (쉼표 구분, 미설정 시 로컬만 허용)
CORS_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    name: str = Field(..., description="사용자 이름")
    gender: str = Field(..., description="성별 (M 또는 F)")
    birth_date: str = Field(..., description="생년월일 (YYYY-MM-DD)")
    birth_time: str = Field(..., description="출생시간 (HH:MM)")
    birth_type: str = Field("양력", description="양력 또는 음력")
    is_leap_month: bool = Field(False, description="음력 윤달 여부")

    @field_validator("gender")
    @classmethod
    def check_gender(cls, v: str) -> str:
        if v not in ("M", "F"):
            raise ValueError("성별은 M 또는 F만 가능합니다.")
        return v

    @field_validator("birth_type")
    @classmethod
    def check_birth_type(cls, v: str) -> str:
        if v not in ("양력", "음력"):
            raise ValueError("달력 구분은 '양력' 또는 '음력'만 가능합니다.")
        return v

    @field_validator("birth_date")
    @classmethod
    def check_birth_date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("생년월일은 YYYY-MM-DD 형식이어야 합니다.")
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("유효하지 않은 날짜입니다. (년/월/일 수치를 확인하세요)")
        return v

    @field_validator("birth_time")
    @classmethod
    def check_birth_time(cls, v: str) -> str:
        if not re.fullmatch(r"\d{2}:\d{2}", v):
            raise ValueError("출생시간은 HH:MM 형식이어야 합니다.")
        hh, mm = map(int, v.split(":"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError("유효하지 않은 출생시간입니다.")
        return v

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) > 30:
            raise ValueError("이름은 30자 이내로 입력해주세요.")
        return v

# 🔴 [중요 2] 조선 왕실 명리학 카운슬러 전용 마스터 프롬프트
SYSTEM_MASTER_PROMPT = """
[SYSTEM: 조선시대 왕실 명리학 카운슬러 전용 심층 지식 및 분량 확장 규칙]

너는 전달받은 JSON 데이터를 바탕으로 사용자에게 사주 풀이를 제공하는 '조선 최고 권위의 왕실 국사(國師)이자 고풍스러운 선비 점술가'다.
이 데이터는 이미 정밀 만세력으로 완벽하게 산출된 결과값이므로, 절대로 데이터를 임의로 수정하거나 재계산하지 않고 있는 그대로 깊이 있게 해석에만 사용해라.

[분량 및 서술 심화 규칙]
0. [최우선] 사주 전체 총평 요약 필수 작성:
   - 문서의 가장 첫 번째 응답 섹션으로 반드시 '## 📜 선비님의 사주 총평 요약'을 작성해라.
   - 사주 전체를 관통하는 핵심 기운과 인생의 큰 흐름, 총체적인 운명의 방향성을 3~4줄로 명확하고 인상 깊게 요약해라.
1. 물상론(物象論)의 적극 활용: 천간과 지지를 자연물에 비유하여 선비님이 자신의 사주를 한 편의 그림처럼 상상할 수 있게 묘사해라.
2. 입체적 분석: 일간(나) 하나만 보지 말고, 반드시 '일간 + 월지 + 가장 강한 십신과 오행'을 결합하여 분석해라.
3. 구체적인 예시 제공: 현대적인 직업군과 구체적인 재테크 방식을 3가지 이상 콕 집어 제안해라.
4. 개운법(솔루션) 필수 포함: 각 섹션의 끝에는 결핍된 기운을 보완하고 운을 틔울 수 있는 구체적인 행동 지침(💡)을 추가해라.
5. 분량 강제: 각 섹션은 최소 3~4개의 상세한 문단으로 구성하며, 소제목과 글머리 기호(-)를 적절히 섞어 가독성을 높여라.
6. [핵심 CRO] 소름 돋는 맛보기 훅(Hook) 섹션 필수 생성:
   - 사주 요약표 바로 다음이자 본격적인 풀이 시작 전, 반드시 '### 💡 혹시 이런 경험 있지 않으신가요?'라는 제목의 맛보기 섹션을 강제로 생성해라.
   - 사용자의 사주 특징(일간, 오행의 편중/결핍, 십신 구조 등)을 바탕으로, 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 4가지를 반드시 불릿 포인트(-)로 출력해라.
7. [절대 원칙] 호칭 및 말투:
   - '내담자'나 평범한 호칭은 절대 금지하고, 반드시 '선비님' 또는 이름을 붙인 '선비님'으로 칭하라.
   - 말투는 반드시 조선시대 왕실 점술가의 고풍스럽고 묵직한 어투(~하옵니다, ~하시옵소서, ~하오니, ~함을 경계하시옵소서 등)를 철저히 유지하라.
   - 단정적인 흉언("망합니다" 등)은 금하되, 점술가의 깊이 있는 혜안과 묵직한 조언으로 신뢰를 준다.
   - 고정된 템플릿이나 뻔한 문구를 절대 복사해 쓰지 말고, 제공된 일간, 결핍 오행, 신강/신약 데이터에 기반하여 100% 개인화된 풀이를 작성하라.

[마크다운 출력 포맷]
## 📜 선비님의 사주 총평 요약
(전체적인 운명의 흐름과 삶의 기조를 관통하는 핵심 요약 3~4줄)

## 🔮 선비님의 사주팔자 요약
(사주 원국, 나를 상징하는 기운, 오행의 분포 요약)

### 💡 혹시 이런 경험 있지 않으신가요?
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 1]
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 2]
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 3]
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 4]

### 👤 선비님의 본연의 성향과 잠재력 (일간 + 월지 중심)
(상세 분석 및 💡 마음 개운법)

### 💰 선비님의 재물과 자산의 흐름 (재성, 식상 중심)
(상세 분석 및 💡 재테크 솔루션)

### 💼 선비님의 직업과 사회생활 (관성, 인성 중심)
(상세 분석 및 💡 직장생활 꿀팁)

### ❤️ 선비님의 연애와 인간관계
(상세 분석 및 💡 관계 개운법)

### 🛤️ 선비님의 현재 대운 흐름 (대운 분석)
(상세 분석 및 💡 대운 승부수)

### 🌟 선비님을 위한 종합 카운슬링
(선비님을 위한 고풍스러운 조언과 마무리 축원)
"""

# 🌟 [진짜 만세력 계산 엔진] 사주 원국 및 신강/신약, 결핍 오행 동적 추출
def calculate_saju_engine(user_input: UserInput) -> dict:
    y, m, d = map(int, user_input.birth_date.split('-'))
    hh, mm = map(int, user_input.birth_time.split(':'))
    
    if user_input.birth_type == "음력":
        lunar = Lunar.fromYmdHms(y, m, d, hh, mm, 0, user_input.is_leap_month)
    else:
        solar = Solar.fromYmdHms(y, m, d, hh, mm, 0)
        lunar = solar.getLunar()
        
    bazi = lunar.getEightChar()
    year_stem, year_branch = bazi.getYearGan(), bazi.getYearZhi()
    month_stem, month_branch = bazi.getMonthGan(), bazi.getMonthZhi()
    day_stem, day_branch = bazi.getDayGan(), bazi.getDayZhi()
    hour_stem, hour_branch = bazi.getTimeGan(), bazi.getTimeZhi()

    elements_map = {
        "목": ["甲", "乙", "寅", "卯"],
        "화": ["丙", "丁", "巳", "午"],
        "토": ["戊", "己", "辰", "戌", "丑", "未"],
        "금": ["庚", "辛", "申", "酉"],
        "수": ["壬", "癸", "亥", "子"]
    }
    saju_chars = [year_stem, year_branch, month_stem, month_branch, day_stem, day_branch, hour_stem, hour_branch]
    five_elements = {"wood": 0, "fire": 0, "earth": 0, "metal": 0, "water": 0}
    elements_count = {"목": 0, "화": 0, "토": 0, "금": 0, "수": 0}
    
    for char in saju_chars:
        if char in elements_map["목"]: 
            five_elements["wood"] += 1
            elements_count["목"] += 1
        elif char in elements_map["화"]: 
            five_elements["fire"] += 1
            elements_count["화"] += 1
        elif char in elements_map["토"]: 
            five_elements["earth"] += 1
            elements_count["토"] += 1
        elif char in elements_map["금"]: 
            five_elements["metal"] += 1
            elements_count["금"] += 1
        elif char in elements_map["수"]: 
            five_elements["water"] += 1
            elements_count["수"] += 1

    char_info = {
        "甲": {"hangul": "갑", "element": "목", "hanja": "甲"},
        "乙": {"hangul": "을", "element": "목", "hanja": "乙"},
        "丙": {"hangul": "병", "element": "화", "hanja": "丙"},
        "丁": {"hangul": "정", "element": "화", "hanja": "丁"},
        "戊": {"hangul": "무", "element": "토", "hanja": "戊"},
        "己": {"hangul": "기", "element": "토", "hanja": "己"},
        "庚": {"hangul": "경", "element": "금", "hanja": "庚"},
        "辛": {"hangul": "신", "element": "금", "hanja": "辛"},
        "壬": {"hangul": "임", "element": "수", "hanja": "壬"},
        "癸": {"hangul": "계", "element": "수", "hanja": "癸"},
        "子": {"hangul": "자", "element": "수", "hanja": "子"},
        "丑": {"hangul": "축", "element": "토", "hanja": "丑"},
        "寅": {"hangul": "인", "element": "목", "hanja": "寅"},
        "卯": {"hangul": "묘", "element": "목", "hanja": "卯"},
        "辰": {"hangul": "진", "element": "토", "hanja": "辰"},
        "巳": {"hangul": "사", "element": "화", "hanja": "巳"},
        "午": {"hangul": "오", "element": "화", "hanja": "午"},
        "未": {"hangul": "미", "element": "토", "hanja": "未"},
        "申": {"hangul": "신", "element": "금", "hanja": "申"},
        "酉": {"hangul": "유", "element": "금", "hanja": "酉"},
        "戌": {"hangul": "술", "element": "토", "hanja": "戌"},
        "亥": {"hangul": "해", "element": "수", "hanja": "亥"},
    }

    def get_info(c):
        return char_info.get(c, {"hangul": c, "element": "토", "hanja": c})

    pillars_detail = {
        "year": {"title": "년주", "desc": "태어난 해", "stem": get_info(year_stem), "branch": get_info(year_branch)},
        "month": {"title": "월주", "desc": "태어난 달", "stem": get_info(month_stem), "branch": get_info(month_branch)},
        "day": {"title": "일주", "desc": "나의 중심", "stem": get_info(day_stem), "branch": get_info(day_branch)},
        "hour": {"title": "시주", "desc": "태어난 시간", "stem": get_info(hour_stem), "branch": get_info(hour_branch)}
    }

    # 일간 및 오행 강약, 신강/신약 정밀 판정
    idx = {"甲": "목", "乙": "목", "丙": "화", "丁": "화", "戊": "토", "己": "토", "庚": "금", "辛": "금", "壬": "수", "癸": "수"}
    day_elem = idx.get(day_stem, "토")

    month_elem_map = {
        "寅": ("목", "봄"), "卯": ("목", "봄"), "辰": ("토", "늦봄/환절기"),
        "巳": ("화", "여름"), "午": ("화", "여름"), "未": ("토", "늦여름/환절기"),
        "申": ("금", "가을"), "酉": ("금", "가을"), "戌": ("토", "늦가을/환절기"),
        "亥": ("수", "겨울"), "子": ("수", "겨울"), "丑": ("토", "늦겨울/환절기")
    }
    month_elem, season_str = month_elem_map.get(month_branch, ("토", "환절기"))

    supporting_elem_map = {
        "목": ("수", "목"),
        "화": ("목", "화"),
        "토": ("화", "토"),
        "금": ("토", "금"),
        "수": ("금", "수")
    }
    in_elem, bi_elem = supporting_elem_map.get(day_elem, ("수", "목"))
    support_score = elements_count.get(in_elem, 0) + elements_count.get(bi_elem, 0)
    if month_elem in (in_elem, bi_elem):
        support_score += 1.5

    if support_score >= 4.0:
        day_strength = "신강(身强)"
    elif support_score <= 2.0:
        day_strength = "신약(身弱)"
    else:
        day_strength = "중화(中和)"

    order = [("목", elements_count["목"]), ("화", elements_count["화"]), ("토", elements_count["토"]), ("금", elements_count["금"]), ("수", elements_count["수"])]
    zeros = [k for k, v in order if v == 0]
    if zeros:
        weak_elem = zeros[0]
    else:
        weak_elem = min(order, key=lambda x: x[1])[0]

    strong_elem = max(order, key=lambda x: x[1])[0]

    return {
        "user_info": {"gender": user_input.gender, "birth_type": user_input.birth_type, "name": user_input.name},
        "saju_pillars": {
            "year": {"stem": year_stem, "branch": year_branch},
            "month": {"stem": month_stem, "branch": month_branch},
            "day": {"stem": day_stem, "branch": day_branch},
            "hour": {"stem": hour_stem, "branch": hour_branch}
        },
        "pillars_detail": pillars_detail,
        "day_master": day_stem,
        "day_elem": day_elem,
        "day_strength": day_strength,
        "season_str": season_str,
        "strong_elem": strong_elem,
        "weak_elem": weak_elem,
        "five_elements_count": five_elements,
        "elements_count": elements_count,
        "ten_gods": {"year": {"stem_god": "", "branch_god": ""}, "month": {"stem_god": "", "branch_god": ""}, "day": {"branch_god": ""}, "hour": {"stem_god": "", "branch_god": ""}},
        "daewoon": [],
        "current_age": 30
    }

# -------------------------------------------------------------
# 3. 비상 시 안전 폴백 사주 리포트 생성기 (100% 동적 개인화 엔진 탑재)
# -------------------------------------------------------------
def generate_emergency_saju_report(name: str, saju_data: dict) -> str:
    """
    사용자의 실제 사주 원국(일간 10종, 오행 분포, 신강/신약, 월지 계절)에 맞춰
    5대 섹션을 비롯한 전 6장의 풀이를 100% 동적으로 맞춤 생성하는 조선 왕실 명리학 엔진
    """
    return build_personalized_emergency_report(name, saju_data)

# -------------------------------------------------------------
# 4. 헬스체크 및 사주 분석 API (다단계 Fallback 모델 및 안전 엔진 탑재)
# -------------------------------------------------------------
@app.get("/api/health")
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Deep Saju API"}

CANDIDATE_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro"
]

@app.post("/api/v1/analyze-saju")
@app.post("/v1/analyze-saju")
@app.post("/api/analyze-saju")
@app.post("/analyze-saju")
def analyze_saju(user_input: UserInput, db: Session = Depends(get_db)):
    try:
        final_name = user_input.name.strip() if user_input.name and user_input.name.strip() else "사용자"
        user_input.name = final_name

        saju_json_data = calculate_saju_engine(user_input)
        saju_json_data["user_info"]["name"] = final_name
        
        prompt = build_dynamic_gemini_prompt(final_name, saju_json_data)

        interpretation_text = None
        last_error = None

        # 🌟 1단계: Google Gemini API 키 유효 시 순차 시도 (동적 프롬프트 주입)
        if GEMINI_API_KEY and len(GEMINI_API_KEY.strip()) > 10:
            for model_name in CANDIDATE_MODELS:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name, 
                        system_instruction=SYSTEM_MASTER_PROMPT
                    )
                    response = model.generate_content(prompt)
                    if response and response.text and len(response.text.strip()) > 50:
                        interpretation_text = response.text
                        break
                except Exception as model_err:
                    last_error = model_err
                    err_str = str(model_err)
                    # 인증 실패 시 불필요한 재시도 방지하고 즉시 고품질 맞춤 엔진으로 직행
                    if any(x in err_str for x in ["API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "403"]):
                        break
                    continue

        # 🌟 2단계: 모든 AI 모델이 Quota 소진 등으로 응답 불가할 때 고품질 안전 폴백 가동
        if not interpretation_text:
            interpretation_text = generate_emergency_saju_report(final_name, saju_json_data)
        
        # 🌟 [DB 영구 저장] 분석 결과 및 입력 데이터를 SQLite에 Insert (서버리스 오류 방어)
        history_id = 0
        try:
            history_record = SajuHistory(
                name=user_input.name,
                gender=user_input.gender,
                birth_date=user_input.birth_date,
                birth_time=user_input.birth_time,
                birth_type=user_input.birth_type,
                ai_interpretation=interpretation_text,
                created_at=datetime.now()
            )
            db.add(history_record)
            db.commit()
            db.refresh(history_record)
            history_id = history_record.id
        except Exception as db_err:
            db.rollback()
            print(f"[WARNING] DB 저장 건너뜀 (서버리스 환경): {db_err}")
        
        return {
            "status": "success",
            "interpretation": interpretation_text,
            "elements_count": saju_json_data.get("elements_count", {}),
            "pillars_detail": saju_json_data.get("pillars_detail", {}),
            "day_master": saju_json_data.get("day_master", "甲"),
            "user_info": {
                "name": final_name,
                "gender": user_input.gender,
                "birth_date": user_input.birth_date,
                "birth_time": user_input.birth_time,
                "birth_type": user_input.birth_type
            },
            "history_id": history_id
        }
    except Exception as e:
        db.rollback()
        # 치명적 서버 오류 시에도 가능한 안전 폴백 제공
        try:
            fallback_saju = calculate_saju_engine(user_input)
            fallback_text = generate_emergency_saju_report(user_input.name, fallback_saju)
            return {
                "status": "success",
                "interpretation": fallback_text,
                "elements_count": fallback_saju.get("elements_count", {}),
                "pillars_detail": fallback_saju.get("pillars_detail", {}),
                "day_master": fallback_saju.get("day_master", "甲"),
                "user_info": {
                    "name": user_input.name or "사용자",
                    "gender": user_input.gender,
                    "birth_date": user_input.birth_date,
                    "birth_time": user_input.birth_time,
                    "birth_type": user_input.birth_type
                },
                "history_id": 0
            }
        except Exception:
            raise HTTPException(status_code=500, detail="사주 분석 엔진 일시 점검 중입니다. 잠시 후 다시 시도해 주세요.")

# -------------------------------------------------------------
# 5. 테스트용 사주 분석 이력 조회 API (최근 10건)
# -------------------------------------------------------------
@app.get("/api/v1/saju-history")
@app.get("/v1/saju-history")
@app.get("/api/saju-history")
@app.get("/saju-history")
async def get_saju_history(db: Session = Depends(get_db), _admin: None = Depends(verify_admin)):
    """
    DB에 영구 저장된 사주 분석 기록 중 최근 10건을 최신순으로 조회합니다.
    """
    try:
        records = (
            db.query(SajuHistory)
            .order_by(SajuHistory.id.desc())
            .limit(10)
            .all()
        )
        return {
            "status": "success",
            "count": len(records),
            "data": [
                {
                    "id": r.id,
                    "name": r.name,
                    "gender": r.gender,
                    "birth_date": r.birth_date,
                    "birth_time": r.birth_time,
                    "birth_type": r.birth_type,
                    "ai_interpretation": r.ai_interpretation,
                    "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None
                }
                for r in records
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------
# 5. 프론트엔드 정적 파일 서빙
# -------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_file = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_file)

@app.get("/bg-image.png")
async def serve_bg():
    bg_file = os.path.join(BASE_DIR, "bg-image.png")
    return FileResponse(bg_file)

@app.get("/emblem.png")
async def serve_emblem():
    emblem_file = os.path.join(BASE_DIR, "emblem.png")
    return FileResponse(emblem_file)

@app.get("/saju-master.gif")
async def serve_saju_master():
    gif_file = os.path.join(BASE_DIR, "saju-master.gif")
    if os.path.exists(gif_file):
        return FileResponse(gif_file)
    return FileResponse(os.path.join(BASE_DIR, "emblem.png"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

