# tvb-ai 환경 구성 가이드

## 파이프라인 설정 (ONNX 전용)
- 추론 관련 설정은 모두 `tvb-ai/tvb-server/detection/gf1/config.py` 안의 상수로 관리합니다.
- `.env.*` 파일이나 `TVB_ENV`, `TVB_ENV_FILE` 같은 환경변수 기반 주입 로직은 제거되었습니다.
- 필요한 값을 변경하려면 해당 파일을 직접 수정하고, 서비스(uvicorn/worker)를 재시작하면 됩니다.

### 핵심 설정 항목
| 키 | 설명 |
| --- | --- |
| `DET_ONNX_PATH`, `CLS_ONNX_PATH` | 얼굴 검출/판별 ONNX 경로 |
| `DET_ONNX_PROVIDERS`, `CLS_ONNX_PROVIDERS` | onnxruntime provider 우선순위 |
| `CONF`, `FPS`, `CLIP_LEN`, `CLIP_STRIDE`, `ALIGN`, `LAYOUT`, `RGB` | 검출 및 샘플링 파라미터 |
| `MEAN`, `STD` | 분류기 입력 정규화 값 |
| `THRESHOLD`, `HIGH_CONF`, `AGGREGATOR`, `TOPK_RATIO`, `TRIM_RATIO` | 판정/집계 기준 |
| `CROP_MARGIN`, `DISABLE_ALIGN_WARP`, `ATTACH_FACES` | 얼굴 정렬 및 결과 첨부 옵션 |
| `EWMA_ALPHA`, `MIN_FACE`, `MIN_DET_SCORE` | 안정화/필터링 파라미터 |
| `TTA_FLIP`, `TEMP_SCALE`, `ONNX_OUTPUT_PROBS`, `LOG_PREPROC`, `LOG_MODEL_OUTPUT` | 분류기 후처리 및 로깅 |

> Torch 기반 분류기 백엔드는 제거되었으며, 모든 추론은 ONNX 세션으로만 수행됩니다.

### 수정 예시
```python
# tvb-ai/tvb-server/detection/pipeline/config.py
DET_ONNX_PATH = "/mnt/models/scrfd_10g_bnkps.onnx"
CLS_ONNX_PATH = "/mnt/models/deepfake_classifier.onnx"
DET_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CLS_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
THRESHOLD = 0.62
```

### 실행 예시
```bash
uvicorn tvb-ai.tvb-server.app:app --reload
python3 tvb-ai/tvb-server/worker.py
```

## RabbitMQ / AWS MQ 설정
FastAPI 및 워커는 `RABBITMQ_URL` 과 관련 플래그를 통해 브로커에 연결합니다. (이 부분은 기존과 동일하게 환경 변수로 관리합니다.)

- `RABBITMQ_URL`: `amqp://` 또는 `amqps://` 형식의 연결 문자열
- `RABBITMQ_USE_TLS`: `true/false` (기본값: URL이 `amqps://`이면 자동 활성화)
- `RABBITMQ_CA_FILE`: TLS 검증에 사용할 CA PEM 경로 (미지정 시 시스템 기본 CA 사용)
- `RABBITMQ_CERT_FILE` / `RABBITMQ_KEY_FILE`: 상호 TLS가 필요할 때 클라이언트 인증서/키 경로
- `RABBITMQ_VERIFY_PEER`: `false`로 설정하면 서버 인증서 검증을 비활성화 (기본값: 검증 활성)
- `RABBITMQ_PREFETCH`: 소비자 채널 prefetch 값 (기본 10)

## 동시 처리(Concurrency)
- `TVB_MAX_CONCURRENCY`: 동시에 처리할 분석 작업 수(기본 1). FastAPI에서 큐로 작업을 넘길 때 사용할 수 있으며 필요 시 환경 변수로 설정합니다.

Spring Boot 측은 `SPRING_RABBITMQ_*` 환경 변수(또는 `application-*.yml`)로 동일한 엔드포인트를 바라보도록 맞춰야 합니다. 테스트 프로필을 사용할 경우 `SPRING_RABBITMQ_SSL_ENABLED=true`, `SPRING_RABBITMQ_PORT=5671`, `SPRING_RABBITMQ_TRUST_STORE=/path/to/amazon-ca.jks` 등을 함께 지정하세요.
