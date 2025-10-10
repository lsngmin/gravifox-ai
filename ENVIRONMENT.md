# tvb-ai 환경 구성 가이드

## FastAPI API 설정
- FastAPI 서버 코드는 `api/` 디렉터리로 이동했어요.
- 런타임 설정은 `api/config/runtime.py`의 `RuntimeSettings` 클래스로 관리해요.
- `.env` 파일이나 환경 변수를 통해 값을 주입할 수 있어요.
- 서버 실행은 아래처럼 해요.

```bash
uvicorn api.main:app --reload
```

### 주요 설정 키
| 키 | 기본값 | 설명 |
| --- | --- | --- |
| `FILE_STORE_ROOT` | `/tmp/uploads` | 업로드 파일 저장 경로 |
| `MAX_IMAGE_MB` | `5` | 이미지 업로드 최대 크기(MB) |
| `MAX_VIDEO_MB` | `50` | 비디오 업로드 최대 크기(MB) |
| `FILE_TTL_HOURS` | `24` | 업로드 파일 TTL(시간) |
| `UPLOAD_TOKEN` | 없음 | `X-Upload-Token` 헤더로 전달할 토큰 |
| `CORS_ALLOW_ORIGINS` | `*` | CORS 허용 origin 목록(콤마 구분) |
| `TVB_VIT_RUN_ROOT` | `<repo>/experiments/vit_residual_fusion` | 최신 ViT 실험 루트 |
| `TVB_VIT_RUN_DIR` | 자동 | 특정 실험 디렉터리를 강제로 지정 |
| `TVB_VIT_CHECKPOINT` | `best.pt` | 사용할 체크포인트 이름 |
| `TVB_VIT_DEVICE` | `auto` | PyTorch 디바이스 문자열 |
| `MODEL_CATALOG_PATH` | `<repo>/api/models/catalog.json` | 모델 카탈로그 JSON 경로 |
| `ENABLE_MQ` | `true` | MQ 소비 여부 |
| `RABBITMQ_URL` | 없음 | RabbitMQ 접속 URL (`amqp://` or `amqps://`) |
| `RABBITMQ_USE_TLS` | 자동 | TLS 강제 여부 |
| `RABBITMQ_PREFETCH` | `10` | MQ prefetch 설정 |
| `ANALYZE_EXCHANGE` | `analyze.exchange` | MQ 교환기 |
| `REQUEST_QUEUE` | `analyze.request.fastapi` | MQ 요청 큐 |

`.env` 예시:
```env
FILE_STORE_ROOT=/var/lib/tvb/uploads
UPLOAD_TOKEN=sample-token
TVB_VIT_RUN_ROOT=/mnt/experiments/vit_residual_fusion
TVB_VIT_DEVICE=cuda:0
RABBITMQ_URL=amqps://user:pass@mq.internal:5671/
RABBITMQ_USE_TLS=true
```

## MQ & 워커 연동
- MQ 관련 코드는 `api/services/mq.py`에 정리했어요.
- 백그라운드 워커는 `api/workers/vit_worker.py`에서 ViT 추론을 수행해요.
- `ENABLE_MQ`가 `true`이고 `RABBITMQ_URL`이 설정된 경우 FastAPI 시작 시 MQ 연결을 시도해요.
- 워커 실행 예시는 아래와 같아요.

```bash
python -m api.workers.vit_worker
```

## Vision Transformer 추론 경로
- FastAPI 이미지 추론 엔드포인트는 `experiments/vit_residual_fusion`에서 최신 실행을 자동으로 선택해요.
- `TVB_VIT_RUN_DIR`를 지정하면 특정 실험 폴더를 직접 사용할 수 있어요.
- 체크포인트는 `checkpoints/best.pt`를 기본으로 사용하며, 존재하지 않으면 `last.pt`로 대체해요.

## 테스트
- FastAPI 전용 테스트는 `api/tests/`에 위치해요.
- 아래 명령어로 실행해요.

```bash
pytest api/tests -q
```

## 업로드 파일 관리
- `MediaStorageService`가 TTL 기반 정리 작업을 주기적으로 실행해요.
- 파일 TTL은 `FILE_TTL_HOURS`로 제어하며 0 이하로 설정하면 정리를 비활성화할 수 있어요.
