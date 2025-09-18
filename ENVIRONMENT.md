# tvb-ai 환경 구성 가이드

## 개요
- `tvb-ai/tvb-server/scrfd/pipeline/config.py`가 시작될 때 `.env.{profile}` 파일을 자동으로 읽어 모델 경로를 설정합니다.
- Spring Boot `application-{profile}.yml`처럼 `TVB_ENV` 값으로 사용할 프로필을 선택할 수 있습니다.
- 별도의 경로 파일을 지정하고 싶다면 `TVB_ENV_FILE` 환경 변수로 직접 지정할 수 있습니다.

## 프로필별 사용법
1. **로컬 개발 (기본)**  
   - `TVB_ENV`를 지정하지 않으면 자동으로 `local` 프로필을 사용합니다.  
   - `tvb-ai/.env.local` 파일의 값을 환경에 맞게 수정해 두세요.

2. **프로덕션/기타 환경**  
   - `TVB_ENV=prod` 처럼 실행 환경에서 프로필 이름을 지정하면 `tvb-ai/.env.prod`를 읽습니다.  
   - 필요하다면 동일한 규칙으로 `.env.staging`, `.env.dev` 등 새로운 파일을 만들어 사용할 수 있습니다.

3. **직접 파일 지정**  
   - 특정 경로의 설정 파일을 쓰고 싶다면 `TVB_ENV_FILE=/path/to/custom.env` 를 환경 변수로 지정하세요.

## 적용 예시
```bash
# 로컬 실행 (기본값)
uvicorn tvb-ai.tvb-server.app:app --reload

# prod 프로필 사용
TVB_ENV=prod uvicorn tvb-ai.tvb-server.app:app

# 커스텀 파일 사용
TVB_ENV_FILE=/etc/tvb-ai/envs/edge-device.env uvicorn tvb-ai.tvb-server.app:app
```

## 변수 목록
- `TVB_DETECTOR_ONNX`: SCRFD 얼굴 검출 ONNX 파일 경로
- `TVB_CLASSIFIER_ONNX`: Deepfake 판별 모델 ONNX 파일 경로
- `TVB_SAMPLE_VIDEO`: 샘플 영상 경로(선택)
- `TVB_DETECTOR_PROVIDERS` / `TVB_CLASSIFIER_PROVIDERS`: 각 세션에 적용할 ONNX Runtime provider 목록(쉼표 구분)
- `TVB_ONNX_PROVIDERS`: 위 두 값이 없을 때 공통 기본 provider로 사용됨

### 백엔드 전환(롤백 가능)
- `TVB_CLASSIFIER_BACKEND`: `onnx`(기본) 또는 `torch`
- `TVB_TORCH_EXTRACTOR_CKPT`: Torch Extractor 체크포인트 경로(TorchScript .pt/.pth 권장)
- `TVB_TORCH_MODEL_CKPT`: Torch Classifier 헤드 체크포인트 경로
- `TVB_TORCH_DEVICE`: `cuda` 또는 `cpu` (기본 `cuda`, 가용성에 따라 자동 폴백)
- `TVB_DUAL_RUN`: `1`이면 Torch 사용 시 에러 시 CPU/ONNX로 폴백 (내부 폴백 로직 있음)
- `TVB_TORCH_BUILDER`: (선택) `pkg.module:build_fn` 형태. TorchScript가 아닌 state_dict 체크포인트일 때, 이 빌더가 `(extractor, classifier)` 모듈을 생성하고 우리가 `state_dict`를 로드합니다.

## MINTIME TorchScript 내보내기(추천)
state_dict 체크포인트를 TorchScript로 변환하면 런타임에서 바로 로드됩니다.

1) 빌더 구현(예: `tvb-ai/tvb-server/mintime_builder.py`의 `build()` 구현)
2) 변환 실행:
```bash
python3 tvb-ai/tvb-server/mintime_export.py \
  --extractor-ckpt tvb-ai/tvb-server/MINTIME_XC_Extractor_checkpoint30 \
  --classifier-ckpt tvb-ai/tvb-server/MINTIME_XC_Model_checkpoint30 \
  --out-extractor tvb-ai/tvb-server/mintime_extractor.pt \
  --out-classifier tvb-ai/tvb-server/mintime_classifier.pt \
  --builder mintime_builder:build \
  --clip-len 8 --size 224
```
3) `.env.prod`에 지정:
```ini
TVB_CLASSIFIER_BACKEND=torch
TVB_TORCH_EXTRACTOR_CKPT=/home/smin/tvb/tvb-ai/tvb-server/mintime_extractor.pt
TVB_TORCH_MODEL_CKPT=/home/smin/tvb/tvb-ai/tvb-server/mintime_classifier.pt
TVB_TORCH_DEVICE=cuda
TVB_CLIP_LEN=8
TVB_CLIP_STRIDE=4
TVB_ALIGN=224
TVB_FAKE_IDX_CLIP=1
```

예시 (주석 해제하여 사용):
```
# TVB_CLASSIFIER_BACKEND=torch
# TVB_TORCH_EXTRACTOR_CKPT=/home/smin/tvb/tvb-ai/tvb-server/MINTIME_XC_Extractor_checkpoint30
# TVB_TORCH_MODEL_CKPT=/home/smin/tvb/tvb-ai/tvb-server/MINTIME_XC_Model_checkpoint30
# TVB_TORCH_DEVICE=cuda
# TVB_CLIP_LEN=8
# TVB_CLIP_STRIDE=4
# TVB_ALIGN=224
# TVB_FAKE_IDX_CLIP=1  # 모델 클래스 순서에 맞게 조정
```

필요 시 `.env.{profile}` 파일에 다른 환경 변수도 자유롭게 추가할 수 있습니다.

## RabbitMQ / AWS MQ 설정
FastAPI 및 워커는 `RABBITMQ_URL`과 관련 플래그를 통해 브로커에 연결합니다. `.env.{profile}`에 아래 항목을 추가하면 테스트/운영에서 공통으로 사용할 수 있습니다.

- `RABBITMQ_URL`: `amqp://` 또는 `amqps://` 형식의 연결 문자열 (예: `amqps://user:pass@b-xxx.mq.ap-northeast-2.amazonaws.com:5671/vhost`)
- `RABBITMQ_USE_TLS`: `true/false` (기본값: URL이 `amqps://`이면 자동 활성화)
- `RABBITMQ_CA_FILE`: TLS 검증에 사용할 CA PEM 경로 (미지정 시 시스템 기본 CA 사용)
- `RABBITMQ_CERT_FILE` / `RABBITMQ_KEY_FILE`: 상호 TLS가 필요할 때 클라이언트 인증서/키 경로
- `RABBITMQ_VERIFY_PEER`: `false`로 설정하면 서버 인증서 검증을 비활성화 (기본값: 검증 활성)
- `RABBITMQ_PREFETCH`: 소비자 채널 prefetch 값 (기본 10)

Spring Boot 측은 `SPRING_RABBITMQ_*` 환경 변수(또는 `application-*.yml`)로 동일한 엔드포인트를 바라보도록 맞춰야 합니다. 테스트 프로필을 사용할 경우 `SPRING_RABBITMQ_SSL_ENABLED=true`, `SPRING_RABBITMQ_PORT=5671`, `SPRING_RABBITMQ_TRUST_STORE=/path/to/amazon-ca.jks` 등을 함께 지정하세요.
