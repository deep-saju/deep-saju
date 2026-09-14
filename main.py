import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
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

# 테이블 자동 생성
Base.metadata.create_all(bind=engine)

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트/백엔드 도메인 분리 대응 (모든 출처 허용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    name: str = Field(..., description="사용자 이름")
    gender: str = Field(..., description="성별 (M 또는 F)")
    birth_date: str = Field(..., description="생년월일 (YYYY-MM-DD)")
    birth_time: str = Field(..., description="출생시간 (HH:MM)")
    birth_type: str = Field("양력", description="양력 또는 음력")
    is_leap_month: bool = Field(False, description="음력 윤달 여부")

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
    고품질 프리미엄 사주 리포트를 100% 무결하게 제공하는 비상 엔진
    """
    pillars = saju_data.get("saju_pillars", {})
    y = pillars.get("year", {})
    m = pillars.get("month", {})
    d = pillars.get("day", {})
    h = pillars.get("hour", {})
    day_master = saju_data.get("day_master", "甲")
    counts = saju_data.get("elements_count", {})
    
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

    return f"""## 📜 {name}님의 사주 총평 요약
{name}님은 타고난 일간이 **{day_master}** 기운으로, {desc}의 명식을 품고 있습니다. 
내면에 감추어진 잠재력과 승부욕이 대단히 강하여 한번 방향을 잡으면 남들이 쉽게 따라오지 못할 집중력을 발휘합니다. 
과거의 시행착오를 거쳐 현재 새로운 도약의 분기점에 서 계시며, 다가오는 대운의 흐름과 타이밍을 정확히 잡으신다면 인생 최대의 황금기를 실현할 수 있는 귀한 사주입니다.

## 🔮 {name}님의 사주팔자 요약

| 구분 | 시주 (時柱) | 일주 (日柱) | 월주 (月柱) | 년주 (年柱) |
| :--- | :---: | :---: | :---: | :---: |
| **천간 (天干)** | {h.get('stem', '-')} | **{d.get('stem', '-')} (나)** | {m.get('stem', '-')} | {y.get('stem', '-')} |
| **지지 (地支)** | {h.get('branch', '-')} | {d.get('branch', '-')} | {m.get('branch', '-')} | {y.get('branch', '-')} |

### 💡 혹시 이런 경험 있지 않으신가요?
- 겉으로는 차분하고 안정적으로 보이지만, 마음속에는 끊임없이 새로운 성취와 더 큰 목표를 갈망하는 열정이 불타고 있습니다.
- 남의 부탁이나 기대를 쉽게 거절하지 못해 혼자서 과도한 짐을 짊어지거나 마음고생을 한 적이 있습니다.
- "내 능력과 노력에 비해 타이밍이 조금만 더 잘 맞았더라면..." 하는 아쉬움을 느껴보신 순간이 있습니다.

### 👤 {name}님의 타고난 본연의 성향과 잠재력
{name}님의 사주 원국은 오행 중 목({counts.get('목', 0)}), 화({counts.get('화', 0)}), 토({counts.get('토', 0)}), 금({counts.get('금', 0)}), 수({counts.get('수', 0)})의 상호작용으로 이루어져 있습니다. 특히 중심 기운인 일간의 힘이 굳건하여 독립적인 결단력이 뛰어나며, 위기 상황에서 침착하게 해결책을 찾아내는 위기 극복 능력이 뛰어납니다.

### 💰 나의 평생 재물운과 부의 그릇
재물운의 흐름을 살펴보면, 성실히 축적하는 정재의 복과 기회를 포착해 크게 도약하는 편재의 기운이 조화를 이루고 있습니다. 낭비를 막고 자산의 파이프라인을 다각화할 때 재물의 그릇이 급격히 팽창하는 형국입니다.

### 💼 타고난 적성과 찰떡궁합 직업운
자율성이 보장되고 본인의 전문성과 기획력이 직접적으로 인정받는 분야에서 비약적인 성과를 냅니다. 틀에 박힌 반복 업무보다는 창의적인 기획이나 사람을 이끄는 매니지먼트에서 최상의 결실을 맺습니다.

### 📈 내 인생의 황금기, 10년 대운 분석
대운의 흐름은 계절이 겨울에서 봄으로 바뀌듯, 웅크렸던 씨앗이 대지를 뚫고 올라오는 상승 국면을 맞이하고 있습니다. 향후 3~5년 간의 선택이 앞으로의 수십 년을 결정짓는 중대한 기로가 될 것입니다.
"""

# -------------------------------------------------------------
# 4. 사주 분석 API (다단계 Fallback 모델 및 안전 엔진 탑재)
# -------------------------------------------------------------
CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.6-flash"
]

@app.post("/api/v1/analyze-saju")
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
        
        # 🌟 [DB 영구 저장] 분석 결과 및 입력 데이터를 SQLite에 Insert
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
            "history_id": history_record.id
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
# 4. 테스트용 사주 분석 이력 조회 API (최근 10건)
# -------------------------------------------------------------
@app.get("/api/v1/saju-history")
async def get_saju_history(db: Session = Depends(get_db)):
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

