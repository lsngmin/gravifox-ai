코딩 규칙 (필수 준수)
[네이밍]

함수: snake_case, 사람이 한눈에 역할을 이해할 수 있는 “동사+목적어” 형태
예) load_model_from_checkpoint, generate_sns_augmentations, run_multiscale_inference
클래스: PascalCase
예) ResidualBranchExtractor, FusionClassifierHead, ViTResidualFusionModel
변수: 의미가 명확한 이름
예) image_tensor, patch_scores, confidence_score
설정 키: 소문자+언더스코어
예) batch_size, img_size, warmup_epochs
[주석/문서화]

모든 파일/클래스/함수에 한국어 docstring 작성 (초심자도 이해 가능하게)
“무엇을/왜 하는지(목적)”, Args(입력), Returns(출력) 포함
내부 주석은 “핵심 단계”에만 간결히 추가 (라인마다 주석 금지)
[로깅]

print() 사용 금지. 반드시 logging 사용.
core/utils/logger.py 에서 전역 로거 설정(logging + coloredlogs), 각 모듈에서는:
from core.utils.logger import get_logger; logger = get_logger(name)
포맷: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
레벨 사용:
INFO: 주요 이벤트(데이터 크기, 에폭 결과, 추론 요약)
DEBUG: 텐서 shape/세부 단계(초기 디버깅 시)
WARNING/ERROR: 예외 상황
“코드를 더럽히지 않게” 하기 위해:
(1) 전역 로거 1회 설정, (2) 함수 시작/종료·핵심 체크포인트에서만 로그 남김
선택: @log_time 데코레이터(실행 시간 로깅) 제공
[예외처리]

초기 테스트 단계: “필수 예외”만 처리
이미지/파일 로드 실패, 체크포인트 로드 실패, 입력 크기/형식 오류 등
불필요한 try/except 남발 금지. 실패 원인 로깅 후 그대로 raise.
예외 처리 블록/임시 방어 로직에는 “# TODO: 필요 없을 시 삭제 가능” 주석을 남길 것
(추후 안정화 시 제거가 쉽도록)

========================================
[중요] 구현 완료 후 “한국어 단계별 설명”을 함께 제공할 것
설명 대상: 방금 구현한 모든 항목(파일/클래스/함수/설정/실행 방법)
형식: 아래 템플릿을 따라 “단계별로, 구체적이고, 처음 보는 사람도 이해하기 쉽게” 작성
말투: 불필요한 전문용어 최소화, 핵심을 짧고 명확하게
[설명 템플릿]
전체 개요

무엇을 구현했고, 왜 이렇게 구성했는지 한 문단 요약
디렉토리/파일별 역할

어떤 상황을 잡고, 왜 그 정도만 처리했는지
“# TODO: 필요 없을 시 삭제 가능” 위치 안내
유지보수/확장 포인트

새로운 모델 추가(파일+@register), 증강 강도 바꾸는 법, 임계값 조정
로깅 레벨 조정법(INFO/DEBUG), 과다 로그 방지 팁
[간단 예시 문구]

“SNS에서 이미지가 여러 번 압축/리사이즈되는 현실을 학습에 반영하기 위해 ‘증강 연쇄’를 넣었습니다. 이로 인해 운영 환경과 학습 분포의 차이를 줄여 실전 정확도를 높일 수 있습니다.”
“추론은 기본(single)과 정밀(multi) 모드를 지원합니다. multi 모드는 패치를 여러 개 뽑아 평균을 내므로 시간이 더 걸리지만, 품질이 낮은 이미지에서도 안정적인 결과를 기대할 수 있습니다.”