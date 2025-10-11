# Gravifox TVB-AI FastAPI 명세

## 개요
- **베이스 URL**: `/`
- **OpenAPI 제목**: `Gravifox TVB-AI API`
- **버전**: `1.0.0`
- **CORS**: 설정된 origin 리스트에 대해 `GET/POST/PUT/DELETE/PATCH/OPTIONS` 등 모든 메서드와 헤더 허용. 자격 증명은 비활성화됨.

## 인증 및 헤더
- 기본적으로 사용자 인증은 필요하지 않음.
- 업로드 경로는 Spring이 발급한 RS256 `Upload-Token`을 요구하며, FastAPI는 `/api/.well-known/jwks.json`으로 공개된 키를 사용해 검증함.

## 엔드포인트 상세

### `GET /`
- **설명**: API 헬스 체크 및 상태 확인.
- **성공 응답** `200 OK`
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | `message` | `string` | 고정 값 `"tvb-ai api is running"` |
  | `timestamp` | `number` | UNIX epoch 초 단위 타임스탬프 |
  | `model_ready` | `boolean` | ViT 추론 파이프라인 초기화 여부 |
  | `mq_enabled` | `boolean` | 메시지 큐 연동 설정 여부 |

### `POST /predict/image`
- **설명**: 업로드된 이미지를 기반으로 AI/Real 판별 확률을 반환.
- **요청**: `multipart/form-data`
  | 필드 | 타입 | 필수 | 설명 |
  | --- | --- | --- | --- |
  | `file` | `binary` | ✔ | 분석할 이미지 파일 |
- **성공 응답** `200 OK` (`ImagePredictionResponse`)
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | `timestamp` | `number` | 예측 생성 시각 (UNIX 타임스탬프) |
  | `model_version` | `string` | 사용된 모델 파이프라인 이름 |
  | `latency_ms` | `number` | 추론 소요 시간 (밀리초) |
  | `p_real` | `number` | Real 클래스 확률 |
  | `p_ai` | `number` | AI 생성 클래스 확률 |
  | `confidence` | `number` | 임계값/보정 적용 신뢰도 |
  | `decision` | `string` | AdaptiveThresholdCalibrator 기반 최종 판정 |
  | `class_names` | `array[string]` | 모델의 클래스 라벨 목록 |
  | `probabilities` | `array[number]` | 각 클래스별 확률 분포 |
- **에러 응답**
  - `400 Bad Request`: 이미지 디코딩 실패 시 `{"detail": "invalid image file"}` 반환.

### `POST /upload`
- **설명**: 이미지 또는 동영상을 스토리지에 저장하고 업로드 ID를 발급.
- **요청**: `multipart/form-data`
  | 필드 | 타입 | 필수 | 설명 |
  | --- | --- | --- | --- |
| `file` | `binary` | ✔ | 저장할 미디어 파일 |
| `uploadId` | `string` | ✔ | Spring에서 토큰 발급 시 사용한 uploadId (UUID + 확장자 권장) |
- **헤더**: `Upload-Token` (필수, Spring 발급 JWT)
- **성공 응답** `200 OK` (`UploadResponse`)
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | `uploadId` | `string` | 스토리지에 저장된 리소스 식별자 |
- **에러 응답**
  - `401 Unauthorized`: 토큰 누락/검증 실패 시.
  - `409 Conflict`: 토큰에 포함된 uploadId와 요청 본문이 불일치하거나 이미 사용된 토큰.

### `GET /models`
- **설명**: 사용 가능한 모델 카탈로그 조회.
- **성공 응답** `200 OK` (`ModelListResponse`)
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | `defaultKey` | `string` | 기본 선택 모델 키 |
  | `items` | `array[ModelItem]` | 모델 목록 |

  `ModelItem`
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | `key` | `string` | 모델 키 |
  | `name` | `string` | 모델 이름 |
  | `version` | `string` | 버전 (선택) |
  | `description` | `string` | 설명 (선택) |
  | `type` | `string` | 모델 유형 (예: 이미지/영상) |
  | `input` | `string` | 기대 입력 형식 |
  | `threshold` | `number` | 판정 임계값 |
  | `labels` | `array[string]` | 지원 클래스 라벨 |

### `POST /predict/video`
- **설명**: 영상 판별 요청. 현재 미구현 상태.
- **응답**: 항상 `501 Not Implemented`와 `{"detail": "video inference not implemented yet"}` 반환.

### `POST /explain/heatmap`
- **설명**: 히트맵 기반 설명 요청. 현재 미구현 상태.
- **응답**: 항상 `501 Not Implemented`와 `{"detail": "heatmap explainability not implemented yet"}` 반환.

### `POST /metrics/ingest`
- **설명**: 외부에서 수집한 메트릭 수신. 현재는 단순 수락만 수행.
- **성공 응답** `202 Accepted`
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | `status` | `string` | 고정 값 `"accepted"` |

## 비동기 백그라운드 작업
- 애플리케이션 시작 시:
  - 스토리지 정리 루프 실행.
  - ViT 추론 서비스 초기화.
  - 메시지 큐가 활성화된 경우 `analyze.request` 큐 소비자 시작.
- 애플리케이션 종료 시:
  - 메시지 큐 소비 작업 취소 및 연결 종료.
  - ViT 추론 서비스 종료.

## 스키마 요약
- `ImagePredictionResponse`, `UploadResponse`, `ModelListResponse`는 `pydantic` 모델로 정의되어 있으며 FastAPI 응답 모델에 사용됨.
