🏪 Merchant Analytics Tool-Using Agent

신한카드 가맹점 데이터 기반 MCP Tool-Using Agent — 질문 의도에 따라 분석 도구를 선택적으로 호출해 마케팅 진단 보고서를 생성합니다.

<br>
📌 프로젝트 개요
가맹점 데이터를 기반으로 사용자의 질문 의도를 분석하고, 적절한 도구를 선택적으로 호출해 마케팅 진단 보고서를 자동 생성하는 서비스형 Tool-Using Agent 시스템입니다.
단순 질의응답 챗봇이 아니라, 엔티티 확정 → 분석 도구 호출 → 근거 기반 보고서 생성의 절차형 흐름을 통해 분석 결과의 신뢰도와 재현 가능성을 높였습니다.

개발 기간: 2025.09 ~ 2025.10
역할: MCP 기반 도구 서버 설계, 엔티티 확정 로직, ReAct 절차 제어, 세션 히스토리 최적화
배포: Docker 컨테이너 기반 서비스 배포 버전


Streamlit 기반 로컬 실행 버전은 marketing_agent 레포를 참고하세요.

<br>
🏗️ 시스템 아키텍처
[사용자 질문]
     │
     ▼
[LangGraph ReAct Agent]  ← main.py
  - 질문 의도 분석 (Q1~Q5 태스크 라우팅)
  - MCP 세션을 통해 필요한 도구만 선택 호출
     │
     ▼
[MCP Tool Server]  ← mcp_server.py
  ├── search_merchants              # 부분일치/마스킹 기반 가맹점 검색
  ├── recommend_channels            # 페르소나/A_STAGE 기반 채널 추천
  ├── analyze_low_revisit           # 4P 기반 재방문율 분석
  ├── analyze_main_customer         # 핵심 고객 세그먼트 분석
  ├── analyze_competitive_positioning  # 경쟁 포지셔닝 진단
  └── analyze_q3                    # 종합 문제점 자동 진단
     │
     ▼
[Markdown 보고서 생성]
  요약 → 핵심 인사이트 → 추천 전략 → 사용 데이터 근거
서비스 서버 / 도구 서버 분리 구조

FastAPI 서비스 서버 (main.py): 대화 처리 및 Agent 실행
FastMCP 도구 서버 (mcp_server.py): 검색·분석 기능을 MCP tool로 노출
기능 추가 시 서비스 서버 수정 없이 MCP 도구 단위로 확장 가능

<br>
🔍 핵심 설계 포인트
1. 엔티티 확정을 통한 분석 신뢰도 향상
가맹점 분석은 반드시 특정 매장을 정확히 식별한 뒤 수행되어야 합니다. 사용자가 브랜드명을 불완전하게 입력하거나 여러 후보가 존재할 경우 잘못된 엔티티에 대해 분석이 수행될 위험이 있었습니다.

부분일치·마스킹 검색 기반 가맹점 검색 도구 구현
검색 결과가 여러 개인 경우 후보 리스트 반환 → 사용자가 직접 선택
가맹점 확정 후에만 분석 도구 호출 → 분석 오류 가능성 최소화

2. ReAct 기반 절차형 분석 흐름 제어
자유 생성 방식만으로는 분석 흐름이 흔들리거나 근거 없이 결과를 출력할 가능성이 있었습니다.

시스템 프롬프트에서 가맹점 검색 → 엔티티 확정 → 분석 도구 호출 → 보고서 생성 절차 명시
질문 의도를 Q1~Q5 분석 태스크로 라우팅
보고서 구조 고정: 요약 → 핵심 인사이트 → 추천 전략 및 실행 가이드 → 사용 데이터 근거

3. 실무형 CSV 스키마 대응
실제 제공된 CSV 데이터는 고정된 API 스키마와 달리 컬럼 구조가 달라질 수 있었습니다.

브랜드명 후보 컬럼 자동 선택, 배달 비율 컬럼 추정
문자열 정규화, 퍼센트 변환, A_STAGE 정규화 구현
CSV 컬럼 구성이 일부 달라져도 분석 도구가 동작하는 구조

4. 세션 히스토리 최적화
대화형 Agent는 세션 히스토리가 길어질수록 토큰 사용량이 증가해 응답 지연과 비용이 커질 수 있었습니다.

LangChain RunnableWithMessageHistory로 세션별 히스토리 관리
최근 N개 대화쌍만 유지하는 trim 로직 적용
대화 컨텍스트를 유지하면서도 비용과 지연을 제어

<br>
🛠️ Tech Stack
분류기술언어 / 프레임워크Python, FastAPIAgentLangGraph (ReAct), LangChain (Message History)MCPFastMCPLLMGoogle Gemini데이터 처리Pandas, NumPy인프라Docker
<br>
📂 파일 구조
marketing_agent_docker_version/
├── main.py                  # FastAPI 서버 + ReAct Agent
├── mcp_server.py            # FastMCP 도구 서버 (검색·분석 도구 정의)
├── q2_preprocess.py         # Q2 태스크용 데이터 전처리
├── q2_preprocess_final.py   # Q2 전처리 최종 버전
├── data/                    # 가맹점 CSV 데이터
├── dockerfile               # Docker 컨테이너 설정
├── requirements.txt         # 의존성 목록
└── .python-version          # Python 버전 설정
<br>
🚀 실행 방법
환경 변수 설정
bashGOOGLE_API_KEY=your_google_api_key
Docker로 실행
bashgit clone https://github.com/seojung-lee/marketing_agent_docker_version
cd marketing_agent_docker_version

docker build -t marketing-agent .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_google_api_key marketing-agent
로컬 실행
bashpip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
<br>
👤 Author
이서정 (Lee Seojung)
M.S. in Mathematics, Ewha Womans University
📝 AI 논문 정리 블로그
