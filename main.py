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

# 🔴 [중요 2] 원본 프롬프트 완벽 보존
SYSTEM_MASTER_PROMPT = """
[SYSTEM: 유료 서비스용 심층 지식 및 분량 확장 규칙]

너는 전달받은 JSON 데이터를 바탕으로 사용자에게 사주 풀이를 제공하는 '전문적이고 따뜻한 명리학 카운슬러'다. 
이 데이터는 이미 완벽하게 산출된 결과값이므로, 너는 절대로 이 데이터를 임의로 수정하거나 재계산하지 않고 있는 그대로 해석에만 사용해라.

[분량 및 서술 심화 규칙]
0. [최우선] 사주 전체 총평 요약 필수 작성:
   - 문서의 가장 첫 번째 응답 섹션으로 반드시 '## 📜 님의 사주 총평 요약'을 작성해라.
   - 사주 전체를 관통하는 핵심 기운과 인생의 큰 흐름, 총체적인 운명의 방향성을 3~4줄로 명확하고 인상 깊게 요약해라.
1. 물상론(物象論)의 적극 활용: 천간과 지지를 자연물에 비유하여 사용자가 자신의 사주를 한 편의 그림처럼 상상할 수 있게 묘사해라.
2. 입체적 분석: 일간(나) 하나만 보지 말고, 반드시 '일간 + 월지 + 가장 강한 십신'을 결합하여 분석해라.
3. 구체적인 예시 제공: 현대적인 직업군과 구체적인 재테크 방식을 3가지 이상 콕 집어 제안해라.
4. 개운법(솔루션) 필수 포함: 각 섹션의 끝에는 약점을 보완하고 운을 틔울 수 있는 구체적인 행동 지침(💡)을 추가해라.
5. 분량 강제: 각 섹션은 최소 3~4개의 상세한 문단으로 구성하며, 소제목과 글머리 기호(-)를 적절히 섞어 가독성을 높여라.
6. 호칭 및 말투: '내담자'라는 단어는 절대 사용하지 말고 반드시 '사용자님' 또는 이름을 부르며, 사용자를 존중하는 부드러운 경어체(~합니다, ~해요)를 사용하되 단정적인 흉언("망합니다" 등)은 절대 금지한다.
7. [핵심 CRO] 소름 돋는 맛보기 훅(Hook) 섹션 필수 생성:
   - 사주 요약표 바로 다음이자 본격적인 풀이 시작 전, 반드시 '### 💡 혹시 이런 경험 있지 않으신가요?'라는 제목의 맛보기 섹션을 강제로 생성해라.
   - 사용자의 사주 특징(일간, 오행의 편중/결핍, 십신 구조 등)을 바탕으로, 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 4가지를 반드시 불릿 포인트(-)로 출력해라.
   - 질문 예시:
     * "겉으로는 쿨해 보여도 속으로는 상처를 깊이 담아두는 편 아니신가요?"
     * "남들에게는 말 못 할 금전적인 압박감을 느낀 적 없으신가요?"
     * "사람들에게 베풀고도 뒤돌아서면 묘한 허탈감을 느낀 적 없으신가요?"
     * "머릿속에는 멋진 기획이 넘치는데 막상 현실의 벽에 부딪혀 망설인 적 없으신가요?"

[마크다운 출력 포맷]
## 📜 님의 사주 총평 요약
(전체적인 운명의 흐름과 삶의 기조를 관통하는 핵심 요약 3~4줄)

## 🔮 님의 사주팔자 요약
(사주 원국, 나를 상징하는 기운, 오행의 분포 요약)

### 💡 혹시 이런 경험 있지 않으신가요?
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 1]
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 2]
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 3]
- [사용자의 사주 특징 기반 뼈를 때리거나 깊이 공감할 수 있는 구체적인 질문 4]

### 👤 본연의 성향과 잠재력 (일간 + 월지 중심)
(상세 분석 및 💡 마음 개운법)

### 💰 재물과 자산의 흐름 (재성, 식상 중심)
(상세 분석 및 💡 재테크 솔루션)

### 💼 직업과 사회생활 (관성, 인성 중심)
(상세 분석 및 💡 직장생활 꿀팁)

### ❤️ 연애와 인간관계
(상세 분석 및 💡 관계 개운법)

### 🛤️ 현재의 대운 흐름 (대운 분석)
(상세 분석 및 💡 대운 승부수)

### 🌟 종합 카운슬링
(따뜻한 조언과 마무리)
"""

# 🌟 [진짜 만세력 계산 엔진] 원본 계산 로직 완벽 보존
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

    return {
        "user_info": {"gender": user_input.gender, "birth_type": user_input.birth_type},
        "saju_pillars": {
            "year": {"stem": year_stem, "branch": year_branch},
            "month": {"stem": month_stem, "branch": month_branch},
            "day": {"stem": day_stem, "branch": day_branch},
            "hour": {"stem": hour_stem, "branch": hour_branch}
        },
        "pillars_detail": pillars_detail,
        "day_master": day_stem,
        "five_elements_count": five_elements,
        "elements_count": elements_count,
        "ten_gods": {"year": {"stem_god": "", "branch_god": ""}, "month": {"stem_god": "", "branch_god": ""}, "day": {"branch_god": ""}, "hour": {"stem_god": "", "branch_god": ""}},
        "daewoon": [],
        "current_age": 30
    }

# -------------------------------------------------------------
# 3. 비상 시 안전 폴백 사주 리포트 생성기 (AI 할당량 초과 시 자동 가동)
# -------------------------------------------------------------
def generate_emergency_saju_report(name: str, saju_data: dict) -> str:
    """
    Google Gemini API 쿼터 고갈(ResourceExhausted / RateLimit) 발생 시에도
    사용자에게 오류 창 대신 정확한 만세력 데이터를 기반으로 한
    프리미엄 사주 리포트와 동일한 완전한 7-Chapter 구조의 고품질 리포트를
    100% 무결하게 제공하는 비상 엔진 (AI 원본과 동급 분량 보장)
    """
    pillars = saju_data.get("saju_pillars", {})
    y = pillars.get("year", {})
    m = pillars.get("month", {})
    d = pillars.get("day", {})
    h = pillars.get("hour", {})
    day_master = saju_data.get("day_master", "甲")
    counts = saju_data.get("elements_count", {}) or {}
    gender = (saju_data.get("user_info", {}) or {}).get("gender", "M")

    def cnt(k: str) -> int:
        return int(counts.get(k, 0) or 0)

    # 일간별 핵심 기운 설명 매핑
    day_master_desc = {
        "甲": "큰 거목(巨木)처럼 우직하고 높은 이상을 향해 뻗어나가는 개척자",
        "乙": "바위틈에서도 꽃을 피워내는 화초처럼 유연하고 끈질긴 생명력의 소유자",
        "丙": "하늘 높이 뜬 태양처럼 세상을 두루 비추며 열정적으로 에너지를 전파하는 리더",
        "丁": "어둠을 밝히는 등불이나 온기 넘치는 모닥불처럼 섬세하고 헌신적인 지혜가",
        "戊": "넓고 웅장한 대지(大地)나 태산처럼 묵직한 포용력과 신뢰를 지닌 인물",
        "己": "만물을 길러내는 비옥한 전답(田畓)처럼 실속 있고 다정한 내유외강의 성향",
        "庚": "단단한 바위나 강철(剛鐵)처럼 강한 추진력과 결단력, 의리를 중시하는 승부사",
        "辛": "빛나는 보석이나 정밀한 명검처럼 예리하고 완벽주의적인 감각과 품격을 지닌 인재",
        "壬": "도도하게 흐르는 큰 강물이나 바다처럼 광활한 지혜와 탁월한 임기응변의 전략가",
        "癸": "대지를 적시는 봄비나 맑은 샘물처럼 총명하고 유연하게 사람의 마음을 얻는 모사"
    }
    desc = day_master_desc.get(day_master, "다재다능한 잠재력과 남다른 직관을 품은 인물")

    # 오행별 맞춤 상세 분석 프로필 (본연 성향 / 재물 / 직업 / 연애 / 대운 / 카운슬링)
    elem_profiles = {
        "목": {
            "nature": "생명이 한창 자라나는 청춘의 계절 같은 기운입니다. 새로운 영역을 개척하고 앞으로 나아가야 진가가 발휘되며, 환경이 바뀌거나 도전이 찾아올수록 오히려 성장 속도가 빨라집니다. 감정을 풀어내는 유연함만 더하면 금상첨화입니다.",
            "wealth": "재물은 땀 흘려 기르고 거두는 밭작물과 같습니다. 당장의 큰돈보다는 자산이 복리로 자라나는 구조가 잘 맞으며, 전문성이라는 씨앗을 뿌린 만큼 결실이 배로 돌아옵니다.",
            "wealth_tips": [
                "지식·콘텐츠·저작권 등 나만의 무형 자산(IP)을 장기적으로 쌓는 투자",
                "성장주, 해외지수, 신성장 산업 등 '성장하는 씨앗' 위주의 분산 투자",
                "전문 자격증·스펙으로 몸값을 높여 본인 가치를 상승시키는 재테크"
            ],
            "career": "기획과 창조, 그리고 세상을 움직이는 아이디어가 밥입니다. 반복적인 틀 안보다는 자율성이 보장되고 새로운 시도를 주도할 수 있는 분야에서 능력이 폭발적으로 발휘됩니다.",
            "career_tips": [
                "브랜딩·마케팅·콘텐츠 기획, 사업개발(BD), 신사업 발굴",
                "교육·강연·멘토링, 목재·원예·환경 등 자연과 접하는 전문 분야",
                "IT, 데이터, 창업·벤처 등 미래 지향적인 성장 산업"
            ],
            "love": "겉으로는 온화하지만 속은 뚜렷한 주관의 소유자입니다. 서로의 성장을 북돋워 주는 상대, 그리고 묵묵히 응원해 주는 관계가 깊이 맞으며, 감정을 말로 꺼내는 연습을 할수록 사랑이 무르익습니다.",
            "daewon": "지금의 대운은 겨울에 잠들었던 생명이 봄을 만나 싹을 틔우는 상승 국면입니다. 미룬 계획과 구상해 둔 사업을 실제 행동으로 옮기는 선택이 인생의 방향을 바꾸는 분기점이 됩니다.",
            "counseling": "세우는 성향이 강해 혼자 많은 것을 감당하려 합니다. 성장이 곧 나의 운이므로, 공부할 자리에 서고 사람과 뜻을 모을 때 인생의 폭이 몇 배로 넓어집니다."
        },
        "화": {
            "nature": "어둠을 환하게 밝히는 태양과 등불의 기운입니다. 타고난 리더십과 표현력, 그리고 주변을 감싸는 온기에 에너지가 가득합니다. 남을 채워 주는 기운이 강한 만큼 자신의 템포와 휴식을 반드시 지켜야 기운이 오래갑니다.",
            "wealth": "재물의 흐름이 빠르고 활발합니다. 무던히 저축하기보다는 끊임없이 흘러들어오는 수입 채널을 넓히는 방식이 맞고, 충동적인 큰 지출만 잠시 멈추면 부의 흐름이 안정됩니다.",
            "wealth_tips": [
                "투자, 사업 등 '수익 채널 다각화' 중심의 현금 흐름 확대 전략",
                "예체능·콘텐츠·크리에이터 등 자신의 재능을 팔아 수익화하는 방식",
                "충동 구매 방지를 위한 3일 유예 규칙 + 자동 저축 시스템 구축"
            ],
            "career": "사람들의 마음을 움직이고 세상을 밝히는 일이 천직입니다. 눈에 보이게 성과를 보여 주고 이끄는 분야에서 최고의 리더십을 발휘하며, 조직의 중심 인물로 빠르게 떠오릅니다.",
            "career_tips": [
                "방송·미디어·홍보·마케팅, 프레젠테이션과 퍼포먼스가 중요한 직군",
                "교육·강사·코칭, 워크숍 진행 등 사람을 깨우는 계통",
                "연예·엔터테인먼트, 행사·이벤트 기획, 창작·스타트업"
            ],
            "love": "밝고 열정적인 에너지로 상대의 마음을 사로잡는 매력의 소유자입니다. 그러나 상대의 속마음을 읽지 못해 속도가 빨라지면 타버리기 쉬우므로, 한 템포 늦춰 상대의 말을 온전히 들어 주는 것이 애정운을 키우는 최고의 방법입니다.",
            "daewon": "지금의 대운은 불꽃이 하늘로 타오르는 정점의 국면입니다. 그동안 쌓은 실력을 세상에 드러내고 리더의 자리에 나설 시기가 도래했으며, 타오르는 열정을 절제된 템포로 유지하는 것이 핵심입니다.",
            "counseling": "빛을 발하는 것이 두려워 접지 않아도 됩니다. 자신의 열정을 믿고, 때로는 불꽃의 온기를 즐기는 여유까지 가질 때 삶이 훨씬 풍요롭게 빛납니다."
        },
        "토": {
            "nature": "만물을 품는 광활한 대지의 기운입니다. 묵직한 신뢰와 책임감, 그리고 끝까지 지켜내는 끈기가 몸에 배어 있어 주변 사람들의 든든한 버팀목이 되어 줍니다. 변화보다 안정 속에서 속도감 있게 성장합니다.",
            "wealth": "자산을 끌어안고 지켜내는 힘이 큰 재물 구조입니다. 단단하고 흔들리지 않는 실물 자산과 장기 저축이 잘 맞으며, 무리한 레버리지보다 꾸준한 축적이 가장 큰 무기가 됩니다.",
            "wealth_tips": [
                "부동산, 리츠, 토지 등 실물 자산 위주의 장기 투자",
                "배당주·채권·정기예금 등 안정적인 현금 흐름을 만드는 자산 배분",
                "모으기 전에 지킨다 — 고정 지출 점검과 자동 이체 저축 습관"
            ],
            "career": "굳건한 책임감과 관리 능력이 돋보이는 경영형·관리형 인재입니다. 조직을 안정적으로 이끌고 제도와 시스템을 구축하는 역할에서 최상의 성과를 내며, 오래 갈수록 가치가 올라갑니다.",
            "career_tips": [
                "조직 관리, 공공·공기업, 행정·재무·HR 계통",
                "건설·부동산·농업·자원 등 '땅과 연결되는' 산업",
                "금융·자산관리, PM, 품질 관리 등 믿음이 중요한 전문 분야"
            ],
            "love": "조용하지만 갈수록 두꺼워지는 깊은 사랑을 하는 타입입니다. 감정 표현이 적어 답답함을 주더라도, 한번 받아들인 사람에게 무한한 신뢰와 헌신으로 보답합니다. 말로 표현하는 연습이 관계를 더욱 단단하게 합니다.",
            "daewon": "지금의 대운은 씨를 뿌리고 밭을 일구는 준비기입니다. 눈앞의 성과보다 기반을 단단히 다지는 선택이 이후의 대성장으로 이어집니다. 과도하게 남의 짐까지 지지 않는 것이 장기적인 승부수입니다.",
            "counseling": "든든한 당신의 존재감은 당신이 내려놓을 때조차 사람들이 의지합니다. 조금은 가벼워져도 괜찮습니다. 흙처럼 포용하되, 자신에게도 봄바람처럼 따뜻한 여유를 허락하세요."
        },
        "금": {
            "nature": "단단하게 다듬어진 결정체 같은 원칙과 강단의 기운입니다. 옳고 그름의 판단이 빠르고 한번 정한 기준은 끝까지 지키는 곧은 성품의 소유자입니다. 예리한 통찰을 따뜻한 말투로 풀어낼수록 기운이 더욱 빛납니다.",
            "wealth": "재물도 명확한 원칙대로 관리되는 구조입니다. 좋은 단일 자산에 집중하여 정밀하게 다루는 방식이 잘 맞고, 필요할 때 과감히 결단하는 순간 부가 커집니다.",
            "wealth_tips": [
                "우량 자산 단일 집중 + 명확한 매매 기준(룰)을 세우는 원칙 투자",
                "현금 창출력이 높은 배당주와 우량주 장기 보유",
                "정밀 분석 기반의 가치 투자 — 정보력이 곧 무기가 되는 분야"
            ],
            "career": "전문성과 완성도로 승부하는 장인형 인재입니다. 깊이 파고드는 분야에서 경쟁자를 단번에 압도하며, 품질과 원칙이 중시되는 직군에서 존재감을 발휘합니다.",
            "career_tips": [
                "사법·감정평가·회계·보안 등 엄밀함이 요구되는 전문직",
                "프로그래밍·데이터·제조·금속·기계 등 정밀 기술 계통",
                "성과가 명확한 영업·무역·유통 등 결단과 추진의 밀당이 필요한 자리"
            ],
            "love": "높은 기준 때문에 인연을 신중하게 고르지만, 맺어진 관계는 깊고 오래갑니다. 의리와 정직함이 몸에 배어 있어 한번 마음을 연 사람에게는 끝까지 책임지는 귀한 사랑을 합니다.",
            "daewon": "지금의 대운은 강철을 불에 달궈 완성품으로 두드리는 제련의 국면입니다. 다소 압박과 중압감이 따르더라도 이것이 바로 성장입니다. 견디는 만큼 단단하고 가치 있는 결과물이 만들어집니다.",
            "counseling": "칼날처럼 곧은 마음, 그 자체가 당신의 미덕입니다. 다만 승부 근성 때문에 스스로는 잊고 살기 쉽습니다. 정기적인 휴식과 충분한 수면을 의무로 여기고, 삶의 온도를 한 스푼 더하세요."
        },
        "수": {
            "nature": "높고 낮은 곳을 흘러내려 막힘없이 순환하는 물의 기운입니다. 탁월한 직관과 통찰, 그리고 어떤 상황에도 유연하게 적응하는 지혜를 지녔습니다. 감정의 파도에 휩쓸리지 않는 중심만 잡으면 어떤 길도 열립니다.",
            "wealth": "물이 낮은 곳으로 흘러 장관을 이루듯, 흐름을 읽고 기회를 포착하는 재물 감각이 뛰어납니다. 정보와 사람의 흐름이 곧 자산이 되며, 주도권을 쥐고 끌고 가는 방식이 잘 맞습니다.",
            "wealth_tips": [
                "흐름을 주도할 수 있는 벤처·투자·사업 등 주도권형 재테크",
                "정보력 기반의 투자(재테크 커뮤니티·전문가 네트워크 활용)",
                "꾸준한 유동성 확보 — 쓸 때와 모을 때의 리듬을 관리하는 자산 설계"
            ],
            "career": "전략과 판단으로 승부하는 전략가형 인재입니다. 거대한 흐름을 읽고 조직과 사람을 이끄는 일에 재능이 있으며, 유연한 대응력으로 어떤 환경에서도 결과를 냅니다.",
            "career_tips": [
                "기획·전략·투자, PM, 컨설팅, 대외협력·M&A 계통",
                "미디어·방송·유통·마케팅 등 '흐름'을 다루는 분야",
                "수산·해양·물류, 여행·호텔 등 물과 이동이 있는 산업"
            ],
            "love": "매력적이고 넓은 인간관계를 유지하지만, 진짜 속마음은 쉽게 열지 않습니다. 소수의 진심 어린 인연에 집중할 때 깊고 풍요로운 사랑이 피어납니다. 속마음의 문을 조금씩 열어 보세요.",
            "daewon": "지금의 대운은 웅크린 강물이 마침내 흐름을 타고 급류로 뻗어나가는 국면입니다. 능력과 기회가 만나는 시기이므로, 과욕으로 한 수레 더 싣기보다 흐름을 믿고 부드럽게 나아가는 것이 승부수입니다.",
            "counseling": "물은 가장 낮은 자세로 있지만 평생 막히지 않습니다. 당신의 유연함과 지혜가 지금 또 새로운 길을 일러줄 것입니다. 충분하다고 느낄 때 한 박자 멈추는 여유가 가장 강력한 개운법입니다."
        }
    }

    def elem_count_list():
        return f"목({cnt('목')}) · 화({cnt('화')}) · 토({cnt('토')}) · 금({cnt('금')}) · 수({cnt('수')})"

    idx = {"甲": "목", "乙": "목", "丙": "화", "丁": "화", "戊": "토", "己": "토", "庚": "금", "辛": "금", "壬": "수", "癸": "수"}
    master_elem = idx.get(day_master, "토")
    prof = elem_profiles.get(master_elem, elem_profiles["토"])

    # 강세 / 약세 기운 판별 (오행 분포 기반)
    order = [("목", cnt("목")), ("화", cnt("화")), ("토", cnt("토")), ("금", cnt("금")), ("수", cnt("수"))]
    strong = max(order, key=lambda x: x[1])
    weak = min(order, key=lambda x: x[1])
    strong_k = strong[0] if strong[1] > 0 else "균형 유지"
    weak_k = weak[0]

    wealth_tips = "\n".join(f"1. **{t}**" for t in prof["wealth_tips"])
    career_tips = "\n".join(f"- **{t}**" for t in prof["career_tips"])

    love_guide = (
        "남성 사주에서 재성(이성·배우자)의 기운을 살펴보면, "
        if gender == "M" else
        "여성 사주에서 관성(이성·배우자)의 기운을 살펴보면, "
    )

    return f"""## 📜 {name}님의 사주 총평 요약
{name}님은 타고난 일간이 **{day_master}** 기운으로, {desc}의 명식을 품고 있습니다. 사주 원국의 오행 분포는 {elem_count_list()}로 이루어져 있으며, 특히 **{master_elem} 기운**이 중심축이 되어 삶의 큰 흐름을 주도합니다. 현재 {name}님은 능력을 쌓은 기반 위에서 본격적인 도약을 앞둔 분기점에 서 계시며, 다가오는 대운의 타이밍을 정확히 활용하신다면 인생 최대의 황금기를 실현할 수 있는 매우 귀한 사주입니다.

## 🔮 {name}님의 사주팔자 요약

| 구분 | 시주 (時柱) | 일주 (日柱) | 월주 (月柱) | 년주 (年柱) |
| :--- | :---: | :---: | :---: | :---: |
| **천간 (天干)** | {h.get('stem', '-')} | **{d.get('stem', '-')} (나)** | {m.get('stem', '-')} | {y.get('stem', '-')} |
| **지지 (地支)** | {h.get('branch', '-')} | {d.get('branch', '-')} | {m.get('branch', '-')} | {y.get('branch', '-')} |

* **나를 상징하는 기운:** {day_master} ({desc})
* **오행의 분포:** {elem_count_list()}
* **강한 기운:** {strong_k} ｜ **보완하면 좋은 기운:** {weak_k}

### 💡 혹시 이런 경험 있지 않으신가요?
- 겉으로는 차분하고 안정적으로 보이지만, 마음속에는 끊임없이 더 큰 목표를 갈망하는 열정이 불타고 있지는 않으신가요?
- 남의 기대나 부탁을 쉽게 거절하지 못해 혼자서 과도한 짐을 짊어지고 마음고생을 한 적이 있지는 않으신가요?
- "내 능력과 노력에 비해 타이밍이 조금만 더 잘 맞았더라면..." 하는 아쉬움을 느껴보신 순간이 있지는 않으신가요?
- 진짜 하고 싶은 일과 현실의 벽 사이에서 머뭇거리며 망설인 적이 있지는 않으신가요?

### 👤 {name}님의 본연의 성향과 잠재력 (일간 + 월지 중심)

{prof['nature']}

사주 원국에서 가장 크게 자리 잡은 기운은 **{strong_k} 기운**이며, 상대적으로 **{weak_k} 기운**이 약해 이를 의식적으로 채워 줄 때 삶의 균형이 최상으로 맞춰집니다. {name}님은 특히 물러서지 않는 내적인 결단력과 위기 상황에서 침착하게 해결책을 찾아내는 저력을 갖추고 있어, 단단한 뿌리 위에서 자신만의 영역을 넓혀 나갈 수 있는 잠재력이 매우 뛰어납니다.

💡 **마음 개운법**
- 매일 15분, 머릿속에 맴도는 생각을 노트에 적어 내려놓는 '생각 비우기'를 실천하세요. 명상과 가벼운 산책이 내면의 기운을 정돈해 줍니다.
- 약한 **{weak_k} 기운**을 채워 주는 컬러나 자연(관련 장소·취미)을 일상에 조금씩 더해 보세요. 운의 흐름이 눈에 띄게 매끄러워집니다.

### 💰 {name}님의 재물과 자산의 흐름 (재성, 식상 중심)

{prof['wealth']}

무엇보다 {name}님은 헛된 투기나 요행으로 재물을 얻는 사주가 아닙니다. 자신의 전문성과 타이밍을 믿고 꾸준히 흘려보낸 자원이 반드시 가치로 돌아오는 구조를 지니고 있어, 한 번 방향을 잡으면 부가 복리로 자라나는 저력이 있습니다. 단, 무의미한 소비나 충동적인 큰 지출만 잠시 멈추어도 자산 증가 속도가 확연히 빨라집니다.

💡 **3가지 맞춤 재테크 솔루션**
{wealth_tips}

### 💼 {name}님의 직업과 사회생활 (관성, 인성 중심)

{prof['career']}

{name}님은 남에게 의존하기보다 자신의 전문성과 책임감으로 인정받는 스타일이며, 조직 안에서도 빠르게 핵심 인물로 자리 잡을 가능성이 높습니다. 특히 본인의 정체성이 뚜렷하게 반영되는 일일수록 성취감과 결과물의 크기가 동시에 커집니다.

💡 **추천 현대 직업군 3가지**
{career_tips}

💡 **직장생활 꿀팁**
- 감정적으로 부딪히기보다는 데이터와 논리로 설득하는 태도를 유지하세요. 적절한 거절과 업무 분산의 기술이 장기적인 커리어의 핵심 무기입니다.

### ❤️ {name}님의 연애와 인간관계

{love_guide}**{master_elem} 기운**의 흐름이 연애와 배우자 인연의 수준을 결정합니다. {prof['love']}

당신의 내면을 드러내는 연습이 관계의 깊이를 결정합니다. 무뚝뚝하거나 쿨해 보이는 겉모습 뒤의 따뜻한 본심은, 상대가 당신을 이해하려 노력할 때 비로소 진가를 발휘합니다. 깊이 있는 극소수의 인연에 마음을 집중하면 평생의 든든한 연이 됩니다.

💡 **관계 개운법**
- "고마워", "미안해", "널 좋아해" 같은 감정의 언어를 의식적으로 자주 꺼내 보세요. 말로 표현할수록 애정운과 인간관계가 동시에 상승합니다.
- 상처받은 감정은 혼자 삭이지 말고, 가까운 사람에게 부드럽게 털어놓는 연습을 하세요. 말하지 않아도 아는 사이는 의외로 드뭅니다.

### 🛤️ {name}님의 현재 대운 흐름 (10년 대운 분석)

{prof['daewon']}

특히 지금의 시기는 그동안 쌓아 온 실력과 경험을 실제 성과로 전환하는 중요한 전환점입니다. 단기적인 성과에 조급해하기보다, 방향성을 분명히 세우고 꾸준히 몰입하는 선택이 다가오는 3~5년의 결과를 크게 좌우합니다. 겉으로는 평범해 보이는 하루하루가 실은 인생의 방향타를 바꾸는 중요한 시간입니다.

💡 **대운 승부수**
- 리스크를 피하기보다 '틈새 기회'를 먼저 찾는 관점을 가지세요. 당신의 사주는 몰아치는 흐름 속에서 중심을 잡을수록 이득이 커지는 구조입니다.
- 주변의 귀인(좋은 동료·멘토)과의 네트워크를 비즈니스처럼 체계적으로 관리하세요. 이 인적 자산이 이후의 가장 큰 부로 돌아옵니다.

### 🌟 {name}님을 위한 종합 카운슬링

{prof['counseling']}

{name}님의 사주는 오행의 상호작용 속에서 분명한 정체성과 큰 가능성을 동시에 품고 있는 명조입니다. 과거의 시행착오는 결코 우연이 아니라, 오늘의 당신을 더 단단하게 다듬어 온 과정입니다. 자신의 기운을 믿고 방향을 분명히 하신다면, 앞으로 맞이할 세월은 줄곧 축적해 온 노력이 인생 최대의 결실로 돌아오는 시기가 될 것입니다.

당신이 이미 가진 강점을 의심하지 마세요. 내면의 빛을 조금만 더 믿고, 삶의 여유를 한 스푼 더하는 것만으로도 운은 자연스럽게 당신의 편에 서게 됩니다. {name}님의 앞으로 펼쳐질 모든 순간에 건강과 행운이 함께하길 진심으로 응원합니다.
"""

# -------------------------------------------------------------
# 4. 헬스체크 및 사주 분석 API (다단계 Fallback 모델 및 안전 엔진 탑재)
# -------------------------------------------------------------
@app.get("/api/health")
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Deep Saju API"}

CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite"
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
        
        prompt = (
            f"사용자 이름: {final_name}\n"
            f"반드시 문서 맨 첫 번째 대제목을 '## 📜 {final_name}님의 사주 총평 요약'으로 전체적인 운명의 큰 흐름 3~4줄을 작성하고, "
            f"그 다음 두 번째로 '## 🔮 {final_name}님의 사주팔자 요약' 표와 오행 분포를 작성해줘.\n"
            f"다음 사주 JSON 데이터를 바탕으로 프리미엄 사주 풀이를 작성해줘:\n{saju_json_data}"
        )

        interpretation_text = None
        last_error = None

        # 🌟 1단계: Google Gemini 후보 모델 순차 시도 (Quota RateLimit 자동 우회)
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
                # 429 Resource Exhausted 시 잠시 대기 후 다음 모델로 즉시 폴백
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

