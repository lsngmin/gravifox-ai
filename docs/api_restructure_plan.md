# FastAPI 서버 리네이밍 및 디렉터리 분리 계획

## 1. tvb-server 디렉터리 리네이밍 제안
- **추천 명칭:** `api_service`
  - FastAPI 기반 API 서버만 담당하는 모듈임을 직관적으로 드러냅니다.
  - `core/`, `scripts/` 등과 동일한 level에서 백엔드 인퍼런스 서비스라는 책임을 명시할 수 있습니다.
- **대체 후보:** `gravifox_api`
  - 제품 브랜드와 FastAPI를 동시에 강조하고 싶을 때 사용할 수 있는 선택지입니다.

> **권장 절차**
> 1. `tvb-server` → `api_service` 디렉터리 리네이밍.
> 2. `PYTHONPATH` 또는 모듈 import 경로(`from tvb-server...`)를 일괄 수정.
> 3. `ENVIRONMENT.md`, 배포 스크립트, CI 설정 등에서 해당 경로 문자열을 찾아 업데이트.
> 4. rename 이후 `uvicorn api_service.main:app` 형태로 엔트리포인트 정비.

## 2. FastAPI 서버 디렉터리 구조 개편안
리네이밍 이후 다음과 같은 서브 디렉터리 구성을 권장합니다.

```
api_service/
├── main.py              # FastAPI 인스턴스 생성 및 라우팅 바인딩
├── config/              # 설정 스키마(pydantic BaseSettings)와 환경 분리
│   ├── __init__.py
│   └── runtime.py
├── routes/              # 엔드포인트 정의 (image, video, explain, metrics 등)
│   ├── __init__.py
│   ├── image.py
│   ├── video.py
│   ├── explain.py
│   └── metrics.py
├── dependencies/        # DI/Depends 헬퍼 (모델 로더, MQ 클라이언트 등)
│   ├── __init__.py
│   └── inference.py
├── schemas/             # Pydantic 모델 정의
│   ├── __init__.py
│   ├── image.py
│   └── video.py
├── services/            # 업무 로직 (inference, mq publisher, 파일 관리 등)
│   ├── __init__.py
│   ├── inference.py
│   ├── mq.py
│   └── storage.py
├── workers/             # 백그라운드 소비자/큐 워커
│   ├── __init__.py
│   └── vit_worker.py
└── tests/               # FastAPI 전용 API 테스트 (pytest + httpx)
```

### 마이그레이션 단계
1. `app.py`를 라우팅/서비스/설정 단위로 분리.
   - 엔드포인트 함수는 `routes/` 모듈로 이동.
   - 모델 로드 및 inference 유틸은 `services/inference.py`로 이전.
2. 기존 `settings.py`, `model_registry.py`, `worker.py`, `mq.py` 등은 역할에 따라 `config/`, `services/`, `workers/`로 재배치.
3. `__init__.py`에는 명시적인 `__all__` 작성으로 공개 API를 정의.
4. 루트 레벨에는 `main.py`만 남기고, FastAPI 객체 생성과 미들웨어 등록을 담당하도록 단순화.
5. 테스트 폴더를 추가해 endpoint 단위의 계약을 검증.

## 3. SCRFD 로직 정리 계획
SCRFD 얼굴 감지 모듈은 ViT-B/16 기반 이미지 위조 판별 파이프라인에서 사용하지 않으므로 아래 순서로 정리합니다.

1. **의존성 확인**
   - `detection/` 모듈의 import 사용처를 전수 조사(`rg "detection" api_service` 등).
   - 실제 API 엔드포인트, 워커, 스크립트에서 SCRFD 관련 함수(`preprocess_scrfd`, `run_scrfd`, `decode_scrfd`) 호출 여부를 확인.
2. **기능 대체 여부 검토**
   - ViT-B/16 추론에 필수적인 전처리/후처리와 충돌하거나 의존하지 않는지 확인.
   - 영상용 파이프라인(TimeSformer 등)에서 얼굴 크롭이 필수라면, 대체 전처리(예: torchvision ops, mediapipe)로 교체 계획 수립.
3. **안전한 제거 절차**
   - 사용처가 없다면 `detection/` 디렉터리를 제거하고, 필요 시 README 또는 문서에 Deprecation 기록.
   - 환경 변수(`DET_ONNX_PATH` 등) 및 관련 ONNX 모델 파일 레퍼런스를 삭제.
   - `requirements.txt` 혹은 setup 스크립트에서 SCRFD 전용 의존성(onnxruntime-gpu 등) 제거.
4. **회귀 테스트**
   - 이미지/영상 추론 E2E 테스트를 수행해 SCRFD 삭제 후에도 ViT-B/16 파이프라인이 정상 동작하는지 검증.
   - FastAPI 엔드포인트와 워커 큐가 SCRFD 의존성을 요구하지 않는지 pytest 및 통합테스트로 확인.
5. **마이그레이션 노트 작성**
   - 배포 시점에 SCRFD 제거로 인한 영향 범위, 롤백 방법, 대체 솔루션 등을 CHANGELOG/문서에 기록.

위 계획을 순차적으로 적용하면 FastAPI 서버 모듈이 역할 기반으로 분리되고, 사용하지 않는 SCRFD 자산을 안전하게 정리할 수 있습니다.
