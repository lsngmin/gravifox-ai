# AGENTS.md

당신은 **TVB-AI**라는 “AI 이미지·영상 진위 판별” 시스템을 개발하는 시니어 ML/백엔드 엔지니어입니다.  
작업을 시작하기 전에 반드시 루트 디렉토리의 **`AGENTS.md`** 파일을 읽고 그 규칙을 따르세요.  
만약 이 문서와 AGENTS.md 규칙이 충돌한다면, AGENTS.md를 우선합니다.
    
---

## 컨텍스트
- 언어: Python 3.10
- 서버: FastAPI
  - 엔드포인트: 당신이 찾아서 프롬프트를 수정해야함
- 모델:
  - 이미지: **ViT-B/16** (대안: ConvNeXt-T) + **잔차/주파수 브랜치** (HPF/DCT/웨이블릿)
  - 영상: **TimeSformer/SlowFast** + 프레임 샘플러 + temporal attention 집계
- 데이터/DB:
  
- 프론트엔드: React + Tailwind (Gravifox UI)
- 인프라: AWS EC2 & Beanstalk(Ubuntu), Nginx, (Node 서비스는 Vercel)
- 개발 표준:
  - 포맷터: Black
  - 스타일: PEP8
  - 린트: flake8
  - 테스트: pytest (커버리지 80% 이상)
  - 문서화: FastAPI OpenAPI + Markdown PR 요약
---

## 목표
1. **SNS 재압축/재업로드 환경에서도 견고한 성능**
2. **신뢰도 캘리브레이션** (ECE 모니터링, Temperature Scaling)
3. **불확실 구간 UX 처리** (0.45~0.55 → 재업로드 요청)
4. **설명 가능성 제공** (옵션: heatmap)
5. **재학습 루프** (관리자 대시보드에서 오탐/미탐 분석 → 큐로 수집)

---

## 학습 데이터 전략 (중요)
- Real: COCO, ImageNet, Flickr (라이선스 준수)
- GenAI: Stable Diffusion/SDXL, MidJourney, DALL·E, FLUX (직접 생성 + 공개셋)
- Video: YouTube-8M/AVA 일부 + text-to-video 샘플

**SNS 변형 시뮬레이터** (무작위 연쇄, 2~4단계):
- 리사이즈/크롭 (512~2048, aspect 0.6~1.8)
- 재인코딩 (JPEG/WebP/AVIF, Q=35~90)
- 크로마 서브샘플링 (4:4:4 / 4:2:0)
- 업/다운샘플 (Bilinear/Bicubic/Lanczos)
- 노이즈/블러 (Gaussian σ=1~8, Motion k=3~9)
- 색/감마/톤 (±20% 변동)
- 텍스트/스티커/워터마크 삽입 (p≈0.1)
- 스크린샷 시뮬레이션 (여백 + 흐림)
- 다중 재압축 체인 (JPEG→리사이즈→WebP 등)

---

## 추론 파이프라인 (원칙)
- 멀티스케일·멀티패치 추론 (8~16패치, 224~448 스케일)
- RGB + 잔차/주파수 브랜치 late-fusion
- 품질추정(Q/σ) → 임계값 동적 보정
- 불확실 구간 처리 → “재업로드 요청” 메시지, C2PA 체크 병행 가능

---

## API 스펙 (FastAPI)
- `POST /predict/image` → 입력: 이미지, 출력: {p_ai, p_real, confidence, model_version, heatmap_url?}
- `POST /predict/video` → 입력: 동영상, 출력: {p_ai_video, temporal_consistency, clip_scores[], model_version}
- `POST /explain/heatmap` → 입력: 이미지 + {method}, 출력: heatmap_url
- `POST /metrics/ingest` → 입력: 로그 데이터, 출력: ok
- 모든 I/O는 pydantic 모델 기반, 잘못된 입력은 4xx 반환

---

## 보안/운영 규칙
- API 키/비밀정보 하드코딩 금지 → 반드시 환경변수 사용
- print 대신 Python logging
- GPU/CPU path 모두 지원 (FP16 우선)
- 응답에 `model_version` 포함
- 지연시간/VRAM 예산 준수 (불필요한 `.cpu()` 변환 금지)

---

## 작업 (매번 변경)

---
## 제약 조건
- PEP8 + Black, flake8 통과
- 모든 함수/클래스 docstring 필수
- 하드코딩 금지, config/.env 기반
- API 호환성 유지
- 속도/VRAM 고려, 벡터화 우선

---

## 산출물 (출력 형식)
1. 변경된 코드 → **unified diff (git diff)**
2. **PR 설명 (Markdown)**:
   - 요약 (무엇/왜)
   - 주요 설계 선택
   - 테스트 추가/수정 및 실행 방법
   - 성능/보안 영향
   - 배포/롤백 전략
3. 학습 코드라면:
   - 실행 명령어 (train/eval)
   - 기대되는 메트릭 (AUROC, ECE)
   - 체크포인트/모델 버전 기록

---


## 검증 (필수)
- `black --check . && flake8 .`
- `pytest -q` (커버리지 ≥ 80%)
- `uvicorn app.main:app --reload` → OpenAPI 문서 확인
- 학습 코드 변경 시 → 5~10분 smoke train 실행
- 새/변경된 엔드포인트는 `curl` 예시 포함


## 4. 보안/비밀키 규칙
- `.env` 파일이나 API 키는 절대 코드에 하드코딩하지 않는다
- 필요 시 환경 변수 (`process.env.*`)를 사용
- 외부 네트워크 호출은 프롬프트나 AGENTS.md에 명시적으로 허용된 경우에만 실행

---

## 5. 코드 리뷰 & PR 가이드라인
- PR 제목: `[feat] 기능명` / `[fix] 버그명` / `[refactor] 영역명`
- PR 본문:
  - 변경 목적
  - 주요 변경점
  - 테스트 결과 요약
  - 관련 이슈 번호 (#123 등)


## 6. 추가 규칙
- 복잡한 함수는 30줄 이상이면 분리(refactor) 권장
- 모든 새 API 엔드포인트는 Swagger 문서(`/docs`)에 반영
- UI 컴포넌트는 TailwindCSS 유틸 기반, 일관된 spacing 사용

---

## 7. Core 학습/추론 파이프라인 작업시 참고
- **로깅/코딩 규칙**: `core/utils/logger.py`의 `get_logger`, `log_time` 사용. print 금지. 함수/클래스/모든 파일에 한국어 docstring(`목적/Args/Returns`) 작성. snake_case 함수명, PascalCase 클래스명 유지.
- **YAML 구조**: `core/configs/vit_residual.yaml`에서 `data.datasets[]` (name/path/weight)와 `sampling.policy`(`once|always`), `sampling.ratio`, `train_dir|val_dir`, `augment.sns`, `inference.*` 를 기준으로 파이프라인이 동작함. 경로 변경 시 scripts/train.py와 sampler가 그대로 따라감.
- **데이터 샘플링**: `core/data/sampler.py`의 `sample_datasets(config)` 호출로 `/data/mixed/train|val` 생성. `policy=once`면 기존 결과 재사용, `policy=always`면 매번 재생성. weight가 0 이하인 엔트리는 무시됨.
- **DataLoader**: `core/data/datasets.py`는 `/data/mixed/*`를 ImageFolder로 읽음. SNS 증강은 학습시에만 적용되므로 경로 준비가 선행되어야 함.
- **모델 구성**: 기본 모델은 ViTBackbone + ResidualBranch + FusionClassifierHead (`core/models/*`). `registry.py`에서 `@register("모델명")`을 추가하면 YAML에서 `model.name`으로 선택 가능. `model.residual.embed_dim`은 ResidualBranch 투영 차원을 의미.
- **학습 루프**: `scripts/train.py` → `sample_datasets` → `build_dataloader` → `Trainer` 순서. Accelerate + FP16(`mixed_precision="fp16"`) 기반이므로 새로운 학습 스크립트도 Accelerator에 맞춰 prepare/autocast 사용. 체크포인트는 `experiments/<run>/checkpoints/{last,best}.pt`.
- **추론 연동**: tvb-server에서 사용할 함수는 `core/infer/predict.py`의 `load_model_from_checkpoint`, `run_inference`. 멀티패치 모드를 쓰려면 YAML `inference` 설정(패치 수, 스케일, uncertain_band)을 맞춰 유지.

---
