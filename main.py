import os
import secrets
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
import re
import httpx
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
    DAY_MASTER_META,
    SYSTEM_GUNGHAP_PROMPT,
    build_gunghap_gemini_prompt,
    build_personalized_gunghap_report,
    SYSTEM_TODAY_PROMPT,
    build_today_gemini_prompt,
    build_personalized_today_report,
    extract_today_fortune_data,
    SYSTEM_REUNION_PROMPT,
    build_reunion_gemini_prompt,
    build_personalized_reunion_report,
    estimate_reunion_score
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

# 토스페이(Toss Pay) 결제 연동 환경변수
# ⚠️ API Key는 서버에서만 사용합니다. 프론트엔드/깃에 절대 노출하지 마세요.
TOSS_PAY_API_KEY = os.getenv("TOSS_PAY_API_KEY", "").strip()
TOSS_PAY_API_BASE = "https://pay.toss.im/api/v2"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

# 서버가 신뢰하는 카테고리별 결제 금액 (클라이언트 금액은 절대 신뢰하지 않음)
PAYMENT_PRODUCTS = {
    "saju": {"amount": 7900, "desc": "평생 심층 운명서 리포트"},
    "gunghap": {"amount": 4900, "desc": "두 사람의 백년해로 궁합 심층 비전"},
    "reunion": {"amount": 2900, "desc": "두 사람의 재회 가능성 심층 비전"},
    "today": {"amount": 2200, "desc": "금일 하루 일진 및 승부 비책 리포트"},
}

# -------------------------------------------------------------
# 1. 데이터베이스 세팅 (SQLAlchemy)
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 운영(Vercel)에서는 영구 저장을 위해 외부 Postgres(DATABASE_URL)를 사용한다.
# DATABASE_URL이 없으면 로컬 SQLite를 사용하고, 서버리스에서는 /tmp에 생성한다(재시작 시 초기화).
_RAW_DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("POSTGRES_URL")
    or os.getenv("NEON_DATABASE_URL")
    or ""
).strip()

if _RAW_DATABASE_URL:
    DATABASE_URL = _RAW_DATABASE_URL
    for _prefix in ("postgres://", "postgresql://"):
        if DATABASE_URL.startswith(_prefix):
            DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len(_prefix):]
            break
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
    DB_DRIVER = "postgres"
    DB_PERSISTENT = True
else:
    if os.getenv("VERCEL") or not os.access(BASE_DIR, os.W_OK):
        DB_PATH = os.path.join("/tmp", "saju_database.db")
    else:
        DB_PATH = os.path.join(BASE_DIR, "saju_database.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
    DB_DRIVER = "sqlite"
    DB_PERSISTENT = not bool(os.getenv("VERCEL"))

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

# SQLAlchemy PaymentOrder 모델 정의 (토스페이 주문/결제 상태)
class PaymentOrder(Base):
    __tablename__ = "payment_orders"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_no = Column(String(50), unique=True, index=True, nullable=False)
    category = Column(String(20), nullable=False)
    amount = Column(Integer, nullable=False)
    pay_token = Column(String(50), nullable=True)
    pay_method = Column(String(20), nullable=True)
    # CREATED: 결제 생성됨, PAID: 결제 완료, FAILED: 생성 실패, CANCELLED: 취소
    status = Column(String(20), nullable=False, default="CREATED")
    created_at = Column(DateTime, default=datetime.now)
    paid_at = Column(DateTime, nullable=True)

# SQLAlchemy PurchasedReport 모델 정의 (결제 완료된 리포트 보관함 / 1년 재열람)
class PurchasedReport(Base):
    __tablename__ = "purchased_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_no = Column(String(50), unique=True, index=True, nullable=False)
    access_key = Column(String(80), unique=True, index=True, nullable=False)
    category = Column(String(20), nullable=False)
    amount = Column(Integer, nullable=False)
    customer_name = Column(String(50), nullable=True)
    birth_date = Column(String(20), nullable=True)
    report_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    expires_at = Column(DateTime, nullable=True)

# 테이블 자동 생성 (서버리스 환경에서 예외 발생 시 서비스 중단 방지)
try:
    Base.metadata.create_all(bind=engine)
except Exception as db_init_err:
    print(f"[WARNING] Database initialization skipped or failed: {db_init_err}")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
ADMIN_SESSION_SECRET = (
    os.getenv("ADMIN_SESSION_SECRET", "").strip()
    or os.getenv("PAYMENT_TOKEN_SECRET", "").strip()
    or "saju-admin-session-fallback-secret"
)
ADMIN_COOKIE_NAME = "saju_admin_session"
ADMIN_SESSION_HOURS = 12
ADMIN_LOGIN_WINDOW = 600
ADMIN_LOGIN_MAX_ATTEMPTS = 5
_admin_login_attempts = {}

# 관리자 인증 의존성 (API 키 방식 - 외부 스크립트용)
def verify_admin(x_api_key: Optional[str] = Header(None)):
    """ADMIN_API_KEY 환경변수가 설정된 경우에만 API 키 인증 요구"""
    if ADMIN_API_KEY and x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="관리자 인증이 필요합니다.")


def _admin_session_token(expires_at: int) -> str:
    raw = str(expires_at).encode("utf-8")
    signature = hmac.new(ADMIN_SESSION_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return f"{expires_at}.{signature}"


def _admin_session_valid(token: str) -> bool:
    if not token or "." not in token:
        return False
    expires_raw, signature = token.split(".", 1)
    try:
        expires_at = int(expires_raw)
    except (TypeError, ValueError):
        return False
    if expires_at < int(time.time()):
        return False
    expected = hmac.new(
        ADMIN_SESSION_SECRET.encode("utf-8"), expires_raw.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _request_is_https(request: Request) -> bool:
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "").split(",")[0].strip()
    return proto == "https"


def require_admin(request: Request) -> None:
    """관리자 세션 쿠키가 유효한지 검사합니다."""
    if not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="관리자 비밀번호(ADMIN_PASSWORD)가 서버에 설정되지 않았습니다.",
        )
    token = request.cookies.get(ADMIN_COOKIE_NAME) or ""
    if not _admin_session_valid(token):
        raise HTTPException(status_code=401, detail="관리자 로그인이 필요합니다.")


def require_admin_or_key(request: Request, x_api_key: Optional[str] = Header(None)) -> None:
    """관리자 세션 또는 API 키 중 하나로 인증합니다."""
    if ADMIN_API_KEY and x_api_key and hmac.compare_digest(x_api_key, ADMIN_API_KEY):
        return
    require_admin(request)


def _check_admin_login_rate(ip: str) -> bool:
    now = time.time()
    attempts = [t for t in _admin_login_attempts.get(ip, []) if now - t < ADMIN_LOGIN_WINDOW]
    _admin_login_attempts[ip] = attempts
    return len(attempts) < ADMIN_LOGIN_MAX_ATTEMPTS


def _record_admin_login_failure(ip: str) -> None:
    _admin_login_attempts.setdefault(ip, []).append(time.time())

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

class PartnerInput(BaseModel):
    name: Optional[str] = Field("인연", description="상대방 이름")
    gender: str = Field("F", description="상대방 성별 (M 또는 F)")
    birth_date: str = Field("1996-05-15", description="상대방 생년월일 (YYYY-MM-DD)")
    birth_time: str = Field("12:00", description="상대방 출생시간 (HH:MM)")
    birth_type: str = Field("양력", description="양력 또는 음력")
    is_leap_month: bool = Field(False, description="음력 윤달 여부")

    @field_validator("gender")
    @classmethod
    def check_partner_gender(cls, v: str) -> str:
        if v not in ("M", "F"):
            return "F"
        return v

    @field_validator("birth_type")
    @classmethod
    def check_partner_birth_type(cls, v: str) -> str:
        if v not in ("양력", "음력"):
            return "양력"
        return v

    @field_validator("birth_date")
    @classmethod
    def check_partner_birth_date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            return "1996-05-15"
        return v

class UserInput(BaseModel):
    name: str = Field(..., description="사용자 이름")
    gender: str = Field(..., description="성별 (M 또는 F)")
    birth_date: str = Field(..., description="생년월일 (YYYY-MM-DD)")
    birth_time: str = Field(..., description="출생시간 (HH:MM)")
    birth_type: str = Field("양력", description="양력 또는 음력")
    is_leap_month: bool = Field(False, description="음력 윤달 여부")
    category: Optional[str] = Field("saju", description="카테고리: saju, gunghap, today, reunion")
    partner: Optional[PartnerInput] = Field(None, description="궁합 감정 시 상대방 정보")

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
8. [범위 엄수 - 타 상품과 중복 금지]
   - 이 리포트는 '평생 사주' 전용이다. 오늘 하루의 길흉(일진 운세)이나 특정 상대와의 궁합·재회운은 다루지 말고, 타고난 본원과 평생의 흐름·대운에만 집중하라.
   - '연애와 인간관계' 장은 나 홀로의 애정 성향과 배우자 복(관성·재성)에 한정하고, 특정 상대와의 궁합 점수나 재회 가능성은 절대 언급하지 말라.

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
        lunar_month = -abs(m) if user_input.is_leap_month else m
        try:
            lunar = Lunar.fromYmdHms(y, lunar_month, d, hh, mm, 0)
        except Exception:
            lunar = Lunar.fromYmdHms(y, m, d, hh, mm, 0)
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
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest"
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

        category = (user_input.category or "saju").strip().lower()
        if category not in ("saju", "gunghap", "today", "reunion"):
            category = "saju"

        print(f"[CATEGORY ROUTE] User: '{final_name}', Category parsed: '{category}'")

        partner_resp_info = None
        partner_pillars_detail = None
        partner_elements_count = None
        partner_day_master = None
        today_fortune_data = None

        # -------------------------------------------------------------
        # [카테고리별 조건부 라우팅 및 LLM 프롬프트 분기]
        # -------------------------------------------------------------
        if category == "gunghap":
            print(f"[CATEGORY ROUTE] -> Selected GUNGHAP prompt & system prompt")
            partner = user_input.partner or PartnerInput()
            partner_name = (partner.name or "인연").strip() or "인연"
            partner.name = partner_name
            partner_saju_data = calculate_saju_engine(partner)
            partner_saju_data["user_info"]["name"] = partner_name

            partner_resp_info = {
                "name": partner_name,
                "gender": partner.gender,
                "birth_date": partner.birth_date,
                "birth_time": partner.birth_time,
                "birth_type": partner.birth_type
            }
            partner_pillars_detail = partner_saju_data.get("pillars_detail", {})
            partner_elements_count = partner_saju_data.get("elements_count", {})
            partner_day_master = partner_saju_data.get("day_master", "己")

            prompt = build_gunghap_gemini_prompt(final_name, saju_json_data, partner_name, partner_saju_data)
            current_sys_prompt = SYSTEM_GUNGHAP_PROMPT
            fallback_fn = lambda: build_personalized_gunghap_report(final_name, saju_json_data, partner_name, partner_saju_data)

        elif category == "reunion":
            print(f"[CATEGORY ROUTE] -> Selected REUNION prompt & system prompt")
            partner = user_input.partner or PartnerInput()
            partner_name = (partner.name or "그 사람").strip() or "그 사람"
            partner.name = partner_name
            partner_saju_data = calculate_saju_engine(partner)
            partner_saju_data["user_info"]["name"] = partner_name

            partner_resp_info = {
                "name": partner_name,
                "gender": partner.gender,
                "birth_date": partner.birth_date,
                "birth_time": partner.birth_time,
                "birth_type": partner.birth_type
            }
            partner_pillars_detail = partner_saju_data.get("pillars_detail", {})
            partner_elements_count = partner_saju_data.get("elements_count", {})
            partner_day_master = partner_saju_data.get("day_master", "己")

            prompt = build_reunion_gemini_prompt(final_name, saju_json_data, partner_name, partner_saju_data)
            current_sys_prompt = SYSTEM_REUNION_PROMPT
            fallback_fn = lambda: build_personalized_reunion_report(final_name, saju_json_data, partner_name, partner_saju_data)

        elif category == "today":
            print(f"[CATEGORY ROUTE] -> Selected TODAY prompt & system prompt")
            now = datetime.now()
            solar_today = Solar.fromYmdHms(now.year, now.month, now.day, 12, 0, 0)
            lunar_today = solar_today.getLunar()
            bazi_today = lunar_today.getEightChar()
            today_gan = bazi_today.getDayGan()
            today_zhi = bazi_today.getDayZhi()
            today_ganzhi = f"{today_gan}{today_zhi}"
            year_ganzhi = f"{bazi_today.getYearGan()}{bazi_today.getYearZhi()}"
            today_info = {
                "date_str": now.strftime("%Y년 %m월 %d일"),
                "solar_str": f"양력 {now.year}년 {now.month}월 {now.day}일",
                "day_gan": today_gan,
                "day_zhi": today_zhi,
                "day_ganzhi": today_ganzhi,
                "year_ganzhi": year_ganzhi
            }

            today_fortune_data = extract_today_fortune_data(saju_json_data.get("day_master", "甲"), today_info)

            prompt = build_today_gemini_prompt(final_name, saju_json_data, today_info)
            current_sys_prompt = SYSTEM_TODAY_PROMPT
            fallback_fn = lambda: build_personalized_today_report(final_name, saju_json_data, today_info)

        else:
            print(f"[CATEGORY ROUTE] -> Selected SAJU prompt & system prompt")
            prompt = build_dynamic_gemini_prompt(final_name, saju_json_data)
            current_sys_prompt = SYSTEM_MASTER_PROMPT
            fallback_fn = lambda: generate_emergency_saju_report(final_name, saju_json_data)

        interpretation_text = None
        last_error = None

        # 🌟 1단계: Google Gemini API 키 유효 시 순차 시도 (동적 프롬프트 주입)
        if GEMINI_API_KEY and len(GEMINI_API_KEY.strip()) > 10:
            for model_name in CANDIDATE_MODELS:
                try:
                    print(f"[CATEGORY ROUTE] Calling Gemini model '{model_name}' for category '{category}'...")
                    model = genai.GenerativeModel(
                        model_name=model_name, 
                        system_instruction=current_sys_prompt
                    )
                    response = model.generate_content(
                        prompt,
                        request_options={"timeout": 45}
                    )
                    if response and response.text and len(response.text.strip()) > 50:
                        interpretation_text = response.text
                        print(f"[CATEGORY ROUTE] Gemini model '{model_name}' SUCCESS for category '{category}' (chars={len(interpretation_text)})")
                        break
                except Exception as model_err:
                    last_error = model_err
                    err_str = str(model_err)
                    print(f"[CATEGORY ROUTE] Gemini model '{model_name}' failed: {err_str[:120]}")
                    if any(x in err_str for x in ["API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "403"]):
                        break
                    continue

        # 🌟 2단계: 모든 AI 모델이 Quota 소진 등으로 응답 불가할 때 고품질 안전 폴백 가동
        if not interpretation_text:
            print(f"[CATEGORY ROUTE] Using fallback report generator for category '{category}'")
            interpretation_text = fallback_fn()
        
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
        
        # 궁합 점수 (0~100) 추출 또는 역학 기반 산출
        gunghap_score = None
        if category == "gunghap":
            m = re.search(r'\[궁합점수:\s*(\d{1,3})\s*점?\]', interpretation_text)
            if not m:
                m = re.search(r'궁합\s*점수[는은]?\s*(\d{1,3})\s*점', interpretation_text)
            if not m:
                m = re.search(r'(\d{1,3})\s*점\s*\(', interpretation_text)
            if m:
                try:
                    val = int(m.group(1))
                    if 0 <= val <= 100:
                        gunghap_score = val
                except Exception:
                    pass

            if gunghap_score is None:
                u1_stem = saju_json_data.get("day_master", "甲")
                u2_stem = partner_day_master or "己"
                elem_gen = {"목": "화", "화": "토", "토": "금", "금": "수", "수": "목"}
                elem_kill = {"목": "토", "토": "수", "수": "화", "화": "금", "금": "목"}
                from saju_counseling_engine import DAY_MASTER_META
                e1 = DAY_MASTER_META.get(u1_stem, {}).get("element", "목")
                e2 = DAY_MASTER_META.get(u2_stem, {}).get("element", "토")
                if elem_gen.get(e1) == e2:
                    gunghap_score = 94
                elif elem_gen.get(e2) == e1:
                    gunghap_score = 96
                elif elem_kill.get(e1) == e2:
                    gunghap_score = 82
                elif elem_kill.get(e2) == e1:
                    gunghap_score = 80
                elif e1 == e2:
                    gunghap_score = 88
                else:
                    gunghap_score = 86

        # 재회 점수 (0~100) 추출 또는 역학 기반 산출
        reunion_score = None
        if category == "reunion":
            m = re.search(r'\[재회점수:\s*(\d{1,3})\s*점?\]', interpretation_text)
            if not m:
                m = re.search(r'재회\s*(?:가능성|지수|점수)[는은이]?\s*(\d{1,3})\s*(?:점|%)', interpretation_text)
            if m:
                try:
                    val = int(m.group(1))
                    if 0 <= val <= 100:
                        reunion_score = val
                except Exception:
                    pass

            if reunion_score is None:
                partner_saju_for_score = {
                    "day_master": partner_day_master or "己",
                    "elements_count": partner_elements_count or {},
                    "day_strength": "중화(中和)"
                }
                reunion_score = estimate_reunion_score(saju_json_data, partner_saju_for_score)

        return {
            "status": "success",
            "category": category,
            "interpretation": interpretation_text,
            "gunghap_score": gunghap_score,
            "reunion_score": reunion_score,
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
            "partner_info": partner_resp_info,
            "partner_pillars_detail": partner_pillars_detail,
            "partner_elements_count": partner_elements_count,
            "partner_day_master": partner_day_master,
            "today_data": today_fortune_data,
            "history_id": history_id
        }
    except Exception as e:
        db.rollback()
        # 치명적 서버 오류 시에도 가능한 안전 폴백 제공
        try:
            req_cat = getattr(user_input, "category", "saju") or "saju"
            fallback_saju = calculate_saju_engine(user_input)
            u_name = user_input.name or "사용자"
            fallback_partner_info = None
            fallback_partner_pillars = {}
            fallback_partner_elements = {}
            fallback_partner_dm = "己"
            fallback_gh_score = None
            fallback_reunion_score = None

            if req_cat == "gunghap":
                partner_fallback = getattr(user_input, "partner", None) or PartnerInput()
                p_name = partner_fallback.name or "인연"
                partner_fallback_saju = calculate_saju_engine(partner_fallback)
                fallback_partner_info = {
                    "name": p_name,
                    "gender": partner_fallback.gender,
                    "birth_date": partner_fallback.birth_date,
                    "birth_time": partner_fallback.birth_time,
                    "birth_type": partner_fallback.birth_type
                }
                fallback_partner_pillars = partner_fallback_saju.get("pillars_detail", {})
                fallback_partner_elements = partner_fallback_saju.get("elements_count", {})
                fallback_partner_dm = partner_fallback_saju.get("day_master", "己")
                fallback_text = build_personalized_gunghap_report(u_name, fallback_saju, p_name, partner_fallback_saju)
                m = re.search(r'\[궁합점수:\s*(\d{1,3})\s*점?\]', fallback_text)
                fallback_gh_score = int(m.group(1)) if m else 88
            elif req_cat == "reunion":
                partner_fallback = getattr(user_input, "partner", None) or PartnerInput()
                p_name = partner_fallback.name or "그 사람"
                partner_fallback_saju = calculate_saju_engine(partner_fallback)
                fallback_partner_info = {
                    "name": p_name,
                    "gender": partner_fallback.gender,
                    "birth_date": partner_fallback.birth_date,
                    "birth_time": partner_fallback.birth_time,
                    "birth_type": partner_fallback.birth_type
                }
                fallback_partner_pillars = partner_fallback_saju.get("pillars_detail", {})
                fallback_partner_elements = partner_fallback_saju.get("elements_count", {})
                fallback_partner_dm = partner_fallback_saju.get("day_master", "己")
                fallback_text = build_personalized_reunion_report(u_name, fallback_saju, p_name, partner_fallback_saju)
                m = re.search(r'\[재회점수:\s*(\d{1,3})\s*점?\]', fallback_text)
                fallback_reunion_score = int(m.group(1)) if m else estimate_reunion_score(fallback_saju, partner_fallback_saju)
            elif req_cat == "today":
                now = datetime.now()
                solar_today = Solar.fromYmdHms(now.year, now.month, now.day, 12, 0, 0)
                bazi_today = solar_today.getLunar().getEightChar()
                t_info = {
                    "date_str": now.strftime("%Y년 %m월 %d일"),
                    "solar_str": f"양력 {now.year}년 {now.month}월 {now.day}일",
                    "day_gan": bazi_today.getDayGan(),
                    "day_zhi": bazi_today.getDayZhi(),
                    "day_ganzhi": f"{bazi_today.getDayGan()}{bazi_today.getDayZhi()}",
                    "year_ganzhi": f"{bazi_today.getYearGan()}{bazi_today.getYearZhi()}"
                }
                fallback_text = build_personalized_today_report(u_name, fallback_saju, t_info)
            else:
                fallback_text = generate_emergency_saju_report(u_name, fallback_saju)

            return {
                "status": "success",
                "category": req_cat,
                "interpretation": fallback_text,
                "gunghap_score": fallback_gh_score,
                "reunion_score": fallback_reunion_score,
                "elements_count": fallback_saju.get("elements_count", {}),
                "pillars_detail": fallback_saju.get("pillars_detail", {}),
                "day_master": fallback_saju.get("day_master", "甲"),
                "partner_info": fallback_partner_info,
                "partner_pillars_detail": fallback_partner_pillars,
                "partner_elements_count": fallback_partner_elements,
                "partner_day_master": fallback_partner_dm,
                "user_info": {
                    "name": u_name,
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
async def get_saju_history(db: Session = Depends(get_db), _admin: None = Depends(require_admin_or_key)):
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
# 5-B. 토스페이(Toss Pay) 결제 연동 API
# -------------------------------------------------------------
class TossPaymentCreateInput(BaseModel):
    category: str = Field("saju", description="리포트 카테고리 (saju, gunghap, reunion, today)")


class TossPaymentConfirmInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    order_no: str = Field(..., alias="orderNo", description="결제 생성 시 발급된 주문번호")
    payment_token: str = Field("", alias="paymentToken", description="서버가 서명한 결제 검증 토큰")


def _public_base_url(request: Request) -> str:
    """토스페이가 콜백/복귀에 사용할 공개 URL을 결정합니다."""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    forwarded_host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    proto = forwarded_proto or request.url.scheme or "https"
    host = forwarded_host or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}".rstrip("/")


def _is_local_or_private_url(url: str) -> bool:
    """localhost/사설 IP 여부를 판별합니다. 토스페이 콜백은 외부 공개 URL만 허용합니다."""
    host = (urlparse(url).hostname or "").strip().lower()
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if host.endswith(".local"):
        return True
    if host.startswith("192.168.") or host.startswith("10."):
        return True
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return True
        except (IndexError, ValueError):
            pass
    return False


def _toss_result_callback_url(base_url: str) -> str:
    """로컬 개발에서는 토스가 사설 URL을 거부하므로 데모 콜백으로 대체합니다.
    (로컬에서도 복귀 후 /confirm 상태 조회로 결제를 검증하므로 잠금 해제는 정상 동작)
    """
    if _is_local_or_private_url(base_url):
        return "https://pay.toss.im/payfront/demo/callback"
    return f"{base_url}/api/payments/toss/callback"


def _payment_token_secret() -> str:
    return os.getenv("PAYMENT_TOKEN_SECRET", "").strip() or TOSS_PAY_API_KEY


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _create_payment_token(order_no: str, category: str, amount: int) -> str:
    """주문번호/카테고리/금액을 서버 비밀키로 서명해 위변조를 막습니다."""
    payload = {
        "o": order_no,
        "c": category,
        "a": int(amount),
        "t": int(time.time()),
    }
    body = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signature = hmac.new(
        _payment_token_secret().encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{body}.{_b64url_encode(signature)}"


def _verify_payment_token(token: str) -> Optional[dict]:
    if not token or "." not in token:
        return None
    body, signature = token.split(".", 1)
    expected = hmac.new(
        _payment_token_secret().encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()
    try:
        provided = _b64url_decode(signature)
    except Exception:
        return None
    if not hmac.compare_digest(expected, provided):
        return None
    try:
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    # 결제 검증 토큰 유효기간: 24시간
    try:
        issued_at = int(payload.get("t", 0))
    except Exception:
        return None
    if issued_at < int(time.time()) - 86400:
        return None
    return payload


def _toss_api_request(path: str, payload: dict) -> dict:
    if not TOSS_PAY_API_KEY:
        raise HTTPException(status_code=500, detail="토스페이 API Key가 서버에 설정되지 않았습니다.")
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(f"{TOSS_PAY_API_BASE}{path}", json=payload)
    except httpx.HTTPError as exc:
        print(f"[TOSS PAY] HTTP error on {path}: {exc}")
        raise HTTPException(status_code=502, detail="토스페이 서버와 통신하지 못했습니다. 잠시 후 다시 시도해 주세요.")

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail=f"토스페이 응답을 해석하지 못했습니다. (HTTP {response.status_code})",
        )
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="토스페이 응답 형식이 올바르지 않습니다.")
    return data


def _toss_status(pay_token: Optional[str] = None, order_no: Optional[str] = None) -> dict:
    payload = {"apiKey": TOSS_PAY_API_KEY}
    if pay_token:
        payload["payToken"] = pay_token
    elif order_no:
        payload["orderNo"] = order_no
    else:
        raise HTTPException(status_code=400, detail="결제 상태 조회에 필요한 주문 정보가 없습니다.")

    data = _toss_api_request("/status", payload)
    if data.get("code") != 0:
        message = data.get("msg") or data.get("errorCode") or "결제 상태를 확인하지 못했습니다."
        raise HTTPException(status_code=502, detail=f"토스페이 상태 조회 실패: {message}")
    return data


def _safe_commit(db: Session, context: str) -> bool:
    """서버리스/잠금 환경에서 DB 저장 실패가 결제 흐름을 막지 않도록 best-effort로 커밋합니다."""
    try:
        db.commit()
        return True
    except Exception as exc:
        db.rollback()
        print(f"[TOSS PAY] DB commit skipped ({context}): {exc}")
        return False


def _mark_order_paid(db: Session, order: PaymentOrder, toss_data: dict) -> None:
    order.status = "PAID"
    order.paid_at = datetime.now()
    if toss_data.get("payToken"):
        order.pay_token = str(toss_data.get("payToken"))[:50]
    if toss_data.get("payMethod"):
        order.pay_method = str(toss_data.get("payMethod"))[:20]
    _safe_commit(db, "mark paid")


@app.post("/api/payments/toss/create")
def create_toss_payment(
    payload: TossPaymentCreateInput,
    request: Request,
    db: Session = Depends(get_db),
):
    """서버가 가격을 결정하고 토스페이 결제를 생성합니다. API Key는 응답에 포함하지 않습니다."""
    if not TOSS_PAY_API_KEY:
        raise HTTPException(status_code=500, detail="결제 설정이 완료되지 않았습니다. 관리자에게 문의해 주세요.")

    category = (payload.category or "saju").strip().lower()
    product = PAYMENT_PRODUCTS.get(category)
    if not product:
        raise HTTPException(status_code=400, detail="지원하지 않는 리포트 카테고리입니다.")

    amount = int(product["amount"])
    order_no = f"SAJU-{datetime.now().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
    base_url = _public_base_url(request)

    order = PaymentOrder(
        order_no=order_no,
        category=category,
        amount=amount,
        status="CREATED",
        created_at=datetime.now(),
    )
    db.add(order)
    _safe_commit(db, "create order")

    toss_payload = {
        "apiKey": TOSS_PAY_API_KEY,
        "orderNo": order_no,
        "productDesc": product["desc"],
        "retUrl": f"{base_url}/?toss=return&orderNo={order_no}",
        "retCancelUrl": f"{base_url}/?toss=cancel&orderNo={order_no}",
        "amount": amount,
        "amountTaxFree": 0,
        "autoExecute": True,
        "resultCallback": _toss_result_callback_url(base_url),
        "callbackVersion": "V2",
    }

    try:
        toss_data = _toss_api_request("/payments", toss_payload)
    except HTTPException:
        order.status = "FAILED"
        _safe_commit(db, "mark create failed")
        raise

    if toss_data.get("code") != 0 or not toss_data.get("checkoutPage"):
        order.status = "FAILED"
        _safe_commit(db, "mark create failed")
        message = toss_data.get("msg") or toss_data.get("errorCode") or "결제 생성에 실패했습니다."
        print(f"[TOSS PAY] create failed: {toss_data}")
        raise HTTPException(status_code=502, detail=f"토스페이 결제 생성 실패: {message}")

    order.pay_token = str(toss_data.get("payToken") or "")[:50] or None
    _safe_commit(db, "save pay token")

    return {
        "status": "success",
        "orderNo": order_no,
        "category": category,
        "amount": amount,
        "checkoutPage": toss_data.get("checkoutPage"),
        "paymentToken": _create_payment_token(order_no, category, amount),
    }


@app.post("/api/payments/toss/confirm")
def confirm_toss_payment(payload: TossPaymentConfirmInput, db: Session = Depends(get_db)):
    """복귀한 구매자의 결제를 토스 상태 조회로 검증한 뒤 완료 처리합니다. (멱등)"""
    order_no = (payload.order_no or "").strip()
    token_payload = _verify_payment_token(payload.payment_token)
    if not token_payload or token_payload.get("o") != order_no:
        raise HTTPException(status_code=400, detail="결제 검증 정보가 유효하지 않습니다.")

    category = str(token_payload.get("c") or "")
    product = PAYMENT_PRODUCTS.get(category)
    try:
        token_amount = int(token_payload.get("a") or 0)
    except Exception:
        token_amount = 0
    if not product or int(product["amount"]) != token_amount:
        raise HTTPException(status_code=400, detail="결제 금액 정보가 유효하지 않습니다.")

    order = db.query(PaymentOrder).filter(PaymentOrder.order_no == order_no).first()
    if order and order.category != category:
        raise HTTPException(status_code=400, detail="주문 카테고리가 일치하지 않습니다.")
    if order and int(order.amount) != token_amount:
        raise HTTPException(status_code=400, detail="주문 금액이 일치하지 않습니다.")

    if order and order.status == "PAID":
        return {
            "status": "success",
            "paid": True,
            "orderNo": order_no,
            "category": order.category,
            "amount": order.amount,
            "alreadyPaid": True,
        }

    toss_data = _toss_status(pay_token=(order.pay_token if order else None), order_no=order_no)
    pay_status = toss_data.get("payStatus")
    if pay_status != "PAY_COMPLETE":
        return {
            "status": "pending",
            "paid": False,
            "orderNo": order_no,
            "category": category,
            "amount": token_amount,
            "payStatus": pay_status,
        }

    if str(toss_data.get("orderNo") or "") != order_no:
        raise HTTPException(status_code=400, detail="토스페이 주문번호가 일치하지 않습니다.")
    try:
        paid_total = int(toss_data.get("amount") or 0)
    except Exception:
        paid_total = 0
    if paid_total != token_amount:
        print(f"[TOSS PAY] amount mismatch: order={order_no} expected={token_amount} paid={paid_total}")
        raise HTTPException(status_code=400, detail="결제 금액이 주문 금액과 일치하지 않습니다.")

    if not order:
        # 서버리스 환경에서 DB가 초기화된 경우에도 검증된 결제는 기록을 복구합니다.
        order = PaymentOrder(
            order_no=order_no,
            category=category,
            amount=token_amount,
            status="CREATED",
            created_at=datetime.now(),
        )
        db.add(order)
        _safe_commit(db, "restore order")

    _mark_order_paid(db, order, toss_data)

    return {
        "status": "success",
        "paid": True,
        "orderNo": order_no,
        "category": order.category,
        "amount": order.amount,
        "payMethod": order.pay_method,
    }


@app.post("/api/payments/toss/callback")
async def toss_payment_callback(request: Request, db: Session = Depends(get_db)):
    """토스페이 서버가 보내는 결제 완료 콜백. 상태 조회로 재검증한 뒤 완료 처리합니다."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    order_no = str(body.get("orderNo") or "").strip()
    if not order_no:
        return {"code": 0, "status": "ignored"}

    order = db.query(PaymentOrder).filter(PaymentOrder.order_no == order_no).first()
    if not order:
        print(f"[TOSS PAY] callback for unknown order: {order_no}")
        return {"code": 0, "status": "ignored"}

    if order.status == "PAID":
        return {"code": 0, "status": "already_paid"}

    if body.get("status") != "PAY_COMPLETE":
        return {"code": 0, "status": "ignored"}

    try:
        toss_data = _toss_status(
            pay_token=str(body.get("payToken") or order.pay_token or "") or None,
            order_no=order_no,
        )
    except HTTPException as exc:
        print(f"[TOSS PAY] callback verify failed: {exc.detail}")
        return {"code": 0, "status": "verify_failed"}

    if toss_data.get("payStatus") != "PAY_COMPLETE":
        return {"code": 0, "status": "not_complete"}
    if str(toss_data.get("orderNo") or "") != order_no:
        return {"code": 0, "status": "order_mismatch"}
    try:
        paid_total = int(toss_data.get("amount") or 0)
    except Exception:
        paid_total = 0
    if paid_total != int(order.amount):
        print(f"[TOSS PAY] callback amount mismatch: order={order_no} expected={order.amount} paid={paid_total}")
        return {"code": 0, "status": "amount_mismatch"}

    _mark_order_paid(db, order, toss_data)
    return {"code": 0, "status": "success"}

# -------------------------------------------------------------
# 5-C. 구매 리포트 보관함 (결제 후 1년 재열람) API
# -------------------------------------------------------------
PURCHASE_RETENTION_DAYS = 365


class PurchaseClaimInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    order_no: str = Field(..., alias="orderNo")
    payment_token: str = Field("", alias="paymentToken")
    name: str = Field("", max_length=50)
    birth_date: str = Field("", alias="birthDate", max_length=20)
    report: str = Field("", max_length=300000)


class PurchaseLookupInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    order_no: str = Field(..., alias="orderNo")
    name: str = Field("", max_length=50)


def _purchase_payload(purchase: PurchasedReport) -> dict:
    return {
        "status": "success",
        "orderNo": purchase.order_no,
        "accessKey": purchase.access_key,
        "category": purchase.category,
        "amount": purchase.amount,
        "name": purchase.customer_name,
        "birthDate": purchase.birth_date,
        "report": purchase.report_text,
        "createdAt": purchase.created_at.strftime("%Y-%m-%d %H:%M:%S") if purchase.created_at else None,
        "expiresAt": purchase.expires_at.strftime("%Y-%m-%d") if purchase.expires_at else None,
    }


@app.post("/api/purchases/claim")
def claim_purchase(payload: PurchaseClaimInput, db: Session = Depends(get_db)):
    """결제 검증 토큰으로 본인 확인 후 리포트를 보관함에 저장합니다. (멱등)"""
    order_no = (payload.order_no or "").strip()
    token_payload = _verify_payment_token(payload.payment_token)
    if not token_payload or token_payload.get("o") != order_no:
        raise HTTPException(status_code=400, detail="결제 검증 정보가 유효하지 않습니다.")

    order = db.query(PaymentOrder).filter(PaymentOrder.order_no == order_no).first()
    if not order or order.status != "PAID":
        raise HTTPException(status_code=400, detail="결제 완료된 주문이 아닙니다.")

    report_text = (payload.report or "").strip()
    if not report_text:
        raise HTTPException(status_code=400, detail="저장할 리포트 내용이 없습니다.")

    purchase = db.query(PurchasedReport).filter(PurchasedReport.order_no == order_no).first()
    if not purchase:
        purchase = PurchasedReport(
            order_no=order_no,
            access_key=secrets.token_urlsafe(32),
            category=order.category,
            amount=int(order.amount),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=PURCHASE_RETENTION_DAYS),
        )
        db.add(purchase)

    purchase.report_text = report_text
    purchase.customer_name = (payload.name or "").strip()[:50] or purchase.customer_name
    purchase.birth_date = (payload.birth_date or "").strip()[:20] or purchase.birth_date
    if not purchase.expires_at:
        purchase.expires_at = datetime.now() + timedelta(days=PURCHASE_RETENTION_DAYS)

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[PURCHASE] claim failed: {exc}")
        raise HTTPException(status_code=500, detail="리포트 보관함 저장에 실패했습니다.")

    db.refresh(purchase)
    return _purchase_payload(purchase)


@app.get("/api/purchases/{access_key}")
def get_purchase(access_key: str, db: Session = Depends(get_db)):
    """보관함 링크(access key)로 리포트를 다시 열람합니다."""
    purchase = db.query(PurchasedReport).filter(PurchasedReport.access_key == access_key).first()
    if not purchase:
        raise HTTPException(status_code=404, detail="보관된 리포트를 찾을 수 없습니다.")
    if purchase.expires_at and purchase.expires_at < datetime.now():
        raise HTTPException(status_code=410, detail="열람 기간(1년)이 만료된 리포트입니다.")
    return _purchase_payload(purchase)


@app.post("/api/purchases/lookup")
def lookup_purchase(payload: PurchaseLookupInput, db: Session = Depends(get_db)):
    """주문번호 + 이름으로 다른 기기에서도 보관된 리포트를 찾습니다."""
    order_no = (payload.order_no or "").strip()
    name = (payload.name or "").strip()
    if not order_no or not name:
        raise HTTPException(status_code=400, detail="주문번호와 이름을 모두 입력해 주세요.")

    purchase = db.query(PurchasedReport).filter(PurchasedReport.order_no == order_no).first()
    if not purchase or (purchase.customer_name or "").strip() != name:
        raise HTTPException(status_code=404, detail="일치하는 구매 내역을 찾을 수 없습니다.")
    if purchase.expires_at and purchase.expires_at < datetime.now():
        raise HTTPException(status_code=410, detail="열람 기간(1년)이 만료된 리포트입니다.")
    return _purchase_payload(purchase)


# -------------------------------------------------------------
# 5-D. 관리자 페이지 인증 및 조회 API
# -------------------------------------------------------------
class AdminLoginInput(BaseModel):
    password: str = Field("", max_length=200)


@app.post("/admin/api/login")
def admin_login(payload: AdminLoginInput, request: Request, response: Response):
    if not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=503,
            detail="관리자 비밀번호(ADMIN_PASSWORD)가 서버에 설정되지 않았습니다. Vercel 환경변수를 확인해 주세요.",
        )

    ip = request.client.host if request.client else "unknown"
    if not _check_admin_login_rate(ip):
        raise HTTPException(status_code=429, detail="로그인 시도가 너무 많습니다. 10분 후 다시 시도해 주세요.")

    if not hmac.compare_digest(
        (payload.password or "").encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")
    ):
        _record_admin_login_failure(ip)
        raise HTTPException(status_code=401, detail="비밀번호가 올바르지 않습니다.")

    _admin_login_attempts.pop(ip, None)
    expires_at = int(time.time()) + ADMIN_SESSION_HOURS * 3600
    response.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value=_admin_session_token(expires_at),
        max_age=ADMIN_SESSION_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=_request_is_https(request),
        path="/",
    )
    return {"status": "success", "expiresAt": expires_at}


@app.post("/admin/api/logout")
def admin_logout(response: Response):
    response.delete_cookie(ADMIN_COOKIE_NAME, path="/")
    return {"status": "success"}


@app.get("/admin/api/overview")
def admin_overview(db: Session = Depends(get_db), _admin: None = Depends(require_admin)):
    """관리자 대시보드용 요약/결제/분석/보관함 데이터를 반환합니다."""
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)
    week_start = today_start - timedelta(days=6)

    try:
        orders = db.query(PaymentOrder).order_by(PaymentOrder.id.desc()).limit(100).all()
        paid_orders = (
            db.query(PaymentOrder)
            .filter(PaymentOrder.status == "PAID")
            .order_by(PaymentOrder.id.desc())
            .limit(5000)
            .all()
        )
        reports = db.query(SajuHistory).order_by(SajuHistory.id.desc()).limit(30).all()
        purchases = db.query(PurchasedReport).order_by(PurchasedReport.id.desc()).limit(50).all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DB 조회 실패: {exc}")

    today_paid = [o for o in paid_orders if o.paid_at and o.paid_at >= today_start]
    total_revenue = sum(int(o.amount or 0) for o in paid_orders)

    by_category = {}
    for order in paid_orders:
        bucket = by_category.setdefault(order.category, {"count": 0, "revenue": 0})
        bucket["count"] += 1
        bucket["revenue"] += int(order.amount or 0)

    daily = {}
    for offset in range(7):
        day = (week_start + timedelta(days=offset)).date().isoformat()
        daily[day] = {"date": day, "count": 0, "revenue": 0}
    for order in paid_orders:
        if not order.paid_at:
            continue
        key = order.paid_at.date().isoformat()
        if key in daily:
            daily[key]["count"] += 1
            daily[key]["revenue"] += int(order.amount or 0)

    return {
        "status": "success",
        "serverTime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "database": {
            "driver": DB_DRIVER,
            "persistent": DB_PERSISTENT,
        },
        "stats": {
            "todayRevenue": sum(int(o.amount or 0) for o in today_paid),
            "todayCount": len(today_paid),
            "totalRevenue": total_revenue,
            "totalCount": len(paid_orders),
            "byCategory": by_category,
            "last7Days": list(daily.values()),
        },
        "orders": [
            {
                "orderNo": o.order_no,
                "category": o.category,
                "amount": o.amount,
                "status": o.status,
                "payMethod": o.pay_method,
                "createdAt": o.created_at.strftime("%Y-%m-%d %H:%M:%S") if o.created_at else None,
                "paidAt": o.paid_at.strftime("%Y-%m-%d %H:%M:%S") if o.paid_at else None,
            }
            for o in orders
        ],
        "reports": [
            {
                "id": r.id,
                "name": r.name,
                "gender": r.gender,
                "birthDate": r.birth_date,
                "birthTime": r.birth_time,
                "birthType": r.birth_type,
                "createdAt": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
                "length": len(r.ai_interpretation or ""),
                "preview": (r.ai_interpretation or "")[:120],
            }
            for r in reports
        ],
        "purchases": [
            {
                "orderNo": p.order_no,
                "category": p.category,
                "amount": p.amount,
                "name": p.customer_name,
                "createdAt": p.created_at.strftime("%Y-%m-%d %H:%M:%S") if p.created_at else None,
                "expiresAt": p.expires_at.strftime("%Y-%m-%d") if p.expires_at else None,
                "length": len(p.report_text or ""),
            }
            for p in purchases
        ],
    }


# -------------------------------------------------------------
# 5. 프론트엔드 정적 파일 서빙
# -------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_file = os.path.join(BASE_DIR, "index.html")
    return FileResponse(index_file)

@app.get("/admin")
async def serve_admin():
    admin_file = os.path.join(BASE_DIR, "admin.html")
    if not os.path.exists(admin_file):
        raise HTTPException(status_code=404, detail="관리자 페이지가 준비되지 않았습니다.")
    return FileResponse(
        admin_file,
        headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"},
    )


@app.get("/robots.txt")
async def serve_robots():
    return PlainTextResponse("User-agent: *\nDisallow: /admin\nDisallow: /api/\n")


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

# 프론트엔드에서 사용하는 정적 이미지 (로컬 uvicorn 구동 시 404 방지)
STATIC_ASSET_FILES = {
    "bg.png",
    "icon_saju_new.png",
    "icon_gunghap.png",
    "icon_today_new.png",
    "loading_brush_new.png",
    "emblem.png",
    "bg-image.png"
}

@app.get("/{filename}")
async def serve_static_asset(filename: str):
    if filename in STATIC_ASSET_FILES:
        asset_path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(asset_path):
            return FileResponse(asset_path)
    raise HTTPException(status_code=404, detail="Not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

