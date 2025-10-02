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

## RabbitMQ 설정 (로컬 AI 서버)
FastAPI 및 워커는 `settings.py`의 `RABBITMQ_URL`을 사용합니다. 현재 기본값은 AI 서버(117.17.149.66) 5000 포트, 자격증명 포함으로 설정되어 있습니다.

- `RABBITMQ_URL`: `amqp://gravifox:!Tmdals017217@117.17.149.66:5000/`
- `RABBITMQ_USE_TLS`: URL이 `amqps://`가 아니므로 비TLS. TLS 필요 시 `amqps://`로 교체 후 CA/검증 옵션 적용
- `RABBITMQ_PREFETCH`: 소비자 채널 prefetch 값 (기본 10)

## 동시 처리(Concurrency)
- `TVB_MAX_CONCURRENCY`: 동시에 처리할 분석 작업 수(기본 1). FastAPI에서 큐로 작업을 넘길 때 사용할 수 있으며 필요 시 환경 변수로 설정합니다.

Spring Boot(dev/test)는 `application-dev.yml`, `application-test.yml`에서 AI 서버 MQ(117.17.149.66:5000, TLS 비활성)로 설정되어 있습니다. 필요 시 `SPRING_RABBITMQ_*` 환경 변수로 덮어쓸 수 있습니다.
