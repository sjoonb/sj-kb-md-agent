# 금융분야 마이데이터 표준API 규격

# 제4장. 표준 API 목록

# 4.1 인증 API

□ 고객이 개인신용정보 전송요구 및 본인인증을 수행하기 위해 필요한 API로, 개별인증과 통합인증으로 구분 (상세 명세는 제5장 참조)

# 4.1.1 개별인증 API

|API ID|API 명|
|---|---|
|개별인증-001|인가코드 발급 요청|
|개별인증-002|접근토큰 발급 요청|
|개별인증-003|접근토큰 갱신|
|개별인증-004|접근토큰 폐기|

|URI|HTTP Method|version|industry|resource|
|---|---|---|---|---|
|/oauth/2.0/authorize|GET|해당|해당| |
|/oauth/2.0/token|POST|없음|없음| |
|/oauth/2.0/token|POST|해당|해당| |
|/oauth/2.0/revoke|POST| | | |

# 4.1.2 통합인증 API

|API ID|API 명|
|---|---|
|통합인증-002|접근토큰 발급 요청 (정보제공자 제공)|

|URI|HTTP Method|version|industry|resource|
|---|---|---|---|---|
|/oauth/2.0/token|POST|없음|없음| |

금융보안원 www.fsec.or.kr - 56 -

# 금융분야 마이데이터 표준API 규격

# 4.2 업권별 정보제공 API

□ 고객의 개인신용정보 전송요구에 의거, 정보제공자가 마이데이터사업자에게 개인신용정보를 전송하기 위해 필요한 API (상세 명세는 제6장 참조)

# 4.2.1 공통 (전 업권 또는)

|API ID|API 명|
|---|---|
|공통-001|정보제공- API 목록 조회 (전 업권 공통)|
|공통-002|정보제공- 전송요구 내역 조회 (전 업권 공통)|
|IRP-001|개인형 IRP 계좌 목록 조회 (은행, 금투, 보험 공통)|
|IRP-002|개인형 IRP 계좌 기본정보 조회 (은행, 금투, 보험 공통)|
|IRP-003|개인형 IRP 계좌 추가정보 조회 (은행, 금투, 보험 공통)|
|IRP-004|개인형 IRP 계좌 거래내역 조회 (은행, 금투, 보험 공통)|
|선불-001|선불카드 목록 조회 (은행, 카드 공통)|
|선불-002|선불카드 잔액정보 조회 (은행, 카드 공통)|
|선불-003|선불카드 거래내역 조회 (은행, 카드 공통)|
|선불-004|선불카드 승인내역(결제내역) 조회 (은행, 카드 공통)|

# URI 및 HTTP Method

|version|industry|resource|HTTP Method|정기적 전송주기|
|---|---|---|---|---|
|v1|bank|/apis|GET|기본|
|v1|invest|/consents|GET|기본|
|v1|insu|/irps|GET|기본|
|v1|bank|/irps/basic|POST|기본|
|v1|invest|/irps/detail|POST|추가|
|v1|insu|/irps/transactions|POST|추가|
|v1|bank|/prepaid|GET|기본|
|v1|card|/prepaid/balance|POST|추가|
|v1|bank|/prepaid/transactions|POST|추가|
|v1|card|/prepaid/approval|POST|추가|

금융보안원 www.fsec.or.kr

# 금융분야 마이데이터 표준API 규격

# 인수채권/금전대부 목록

|API ID|API 명|version|URI|HTTP Method|전송주기|
|---|---|---|---|---|---|
|채권-001|조회|v1|/bonds|GET|기본|
|채권-002|잔액정보 조회|v1|/bonds|POST|추가|
|채권-003|거래내역 조회|v1|/bonds/detail|POST|추가|

* 선불-001~선불-004는 카드업권 우선 적용 및 제공하며, 은행업권은 추후 적용 예정

# 4.2.2 은행 업권

|API ID|API 명|version|URI|HTTP Method|전송주기|
|---|---|---|---|---|---|
|은행-001|계좌 목록 조회|v1|/accounts|GET|기본|
|은행-002|수신계좌 기본정보 조회|v1|/accounts/deposit/basic|POST|기본|
|은행-003|수신계좌 추가정보 조회|v1|/accounts/deposit/detail|POST|추가|
|은행-004|수신계좌 거래내역 조회|v1|/accounts/deposit/transactions|POST|추가|
|은행-005|투자상품계좌 기본정보 조회|v1|/accounts/invest/basic|POST|기본|
|은행-006|투자상품계좌 추가정보 조회|v1|/accounts/invest/detail|POST|추가|
|은행-007|투자상품계좌 거래내역 조회|v1|/accounts/invest/transactions|POST|추가|
|은행-008|대출상품계좌 기본정보 조회|v1|/accounts/loan/basic|POST|기본|
|은행-009|대출상품계좌 추가정보 조회|v1|/accounts/loan/detail|POST|추가|
|은행-010|대출상품계좌 거래내역 조회|v1|/accounts/loan/transactions|POST|추가|

금융보안원 www.fsec.or.kr

# 금융분야 마이데이터 표준API 규격

# 4.2.3 카드 업권

|API ID|API 명|version|industry|URI|HTTP Method|정기적 전송주기|
|---|---|---|---|---|---|---|
|카드-001|카드 목록 조회|v1| |/cards|GET|기본|
|카드-002|카드 기본정보 조회|v1| |/cards/{card_id}|GET|기본|
|카드-003|포인트 정보 조회|v1| |/points|GET|추가|
|카드-004|청구 기본정보 조회|v1| |/bills|GET|기본|
|카드-005|청구 추가정보 조회|v1| |/bills/detail|GET|기본|
|카드-006|결제정보 조회|v1| |/payments|GET|추가|
|카드-007|리볼빙 정보 조회|v1|card|/payments/revolving|GET|추가|
|카드-008|국내 승인내역 조회|v1| |/cards/{card_id}/approval-domestic|GET|추가|
|카드-009|해외 승인내역 조회|v1| |/cards/{card_id}/approval-overseas|GET|추가|
|카드-010|대출상품 목록 조회|v1| |/loans|GET|기본|
|카드-011|단기대출 정보 조회|v1| |/loans/short-term|GET|추가|
|카드-012|장기대출 정보 조회|v1| |/loans/long-term|GET|추가|

# 4.2.4 금융투자 업권

|API ID|API 명|version|industry|URI|HTTP Method|정기적 전송주기|
|---|---|---|---|---|---|---|
|금투-001|계좌 목록 조회|v1| |/accounts|GET|기본|
|금투-002|계좌 기본정보 조회|v1| |/accounts/basic|POST|추가|
|금투-003|계좌 거래내역 조회|v1|invest|/accounts/transactions|POST|추가|
|금투-004|계좌 상품정보 조회|v1| |/accounts/products|POST|추가|
|금투-005|연금계좌 추가정보 조회|v1| |/accounts/pension|POST|추가|

금융보안원 www.fsec.or.kr

# 금융분야 마이데이터 표준API 규격

# 4.2.5 보험 업권

|API ID|API 명|
|---|---|
|보험-001|보험 목록 조회|
|보험-002|보험 기본정보 조회|
|보험-003|보험 특약정보 조회|
|보험-004|자동차보험 정보 조회|
|보험-005|보험 납입정보 조회|
|보험-006|보험 거래내역 조회|
|보험-007|자동차보험 거래내역 조회|
|보험-008|대출상품 목록 조회|
|보험-009|대출상품 기본정보 조회|
|보험-010|대출상품 추가정보 조회|
|보험-011|대출상품 거래내역 조회|

|URI|HTTP Method|정기적 전송주기| | |
|---|---|---|---|---|
|v1 /insurances| |GET|기본| |
|v1 /insurances/basic| |POST|기본| |
|v1 /insurances/contracts| |POST|기본| |
|v1 /insurances/car| |POST|기본| |
|v1 /insurances/payment| |POST|기본| |
|v1 /insurances/transactions| |POST|추가| |
|v1 /insurances/car/transactions| |POST|추가| |
|v1 /loans| |GET|기본| |
|v1 /loans/basic| |POST|기본| |
|v1 /loans/detail| |POST|추가| |
|v1 /loans/transactions| |POST|추가| |
|보험-012|보험 보장정보 조회|v1 /insurances/coverages|POST|추가|

# 4.2.6 전자금융 업권

|API ID|API 명|
|---|---|
|전금-001|선불전자지급수단 목록 조회|
|전금-002|선불전자지급수단 잔액정보 조회|
|전금-003|선불전자지급수단 자동충전정보 조회|
|전금-004|선불 거래내역 조회|
|전금-101|계정 목록 조회|
|전금-102|결제수단 등록 정보 조회|
|전금-103|결제내역 조회|

|URI|HTTP Method|정기적 전송주기|
|---|---|---|
|v1 /prepaid|GET|기본|
|v1 /prepaid/balance|POST|추가|
|v1 /prepaid/charge|POST|기본|
|v1 /prepaid/transactions|POST|추가|
|v1 /paid|GET|기본|
|v1 /paid/methods|POST|기본|
|v1 /paid/transactions|POST|추가|

금융보안원 www.fsec.or.kr

# 금융분야 마이데이터 표준API 규격

# 4.2.7 할부금융 업권

|API ID|API 명|version|URI|HTTP Method|정기적 전송주기|
|---|---|---|---|---|---|
|할부금융-001|계좌 목록 조회|v1|/loans|GET|기본|
|할부금융-002|대출상품계좌 기본정보 조회|v1|/loans/basic|POST|기본|
|할부금융-003|대출상품계좌 추가정보 조회|v1|/loans/detail|POST|추가|
|할부금융-004|대출상품계좌 거래내역 조회|v1|/loans/transactions|POST|추가|
|할부금융-005|운용리스 기본정보 조회|v1|/loans/oplease/basic|POST|기본|
|할부금융-006|운용리스 거래내역 조회|v1|/loans/oplease/transactions|POST|추가|

# 4.2.8 보증보험 업권

|API ID|API 명|version|URI|HTTP Method|정기적 전송주기|
|---|---|---|---|---|---|
|보증보험-001|보증보험 목록 조회|v1|/insurances|GET|기본|
|보증보험-002|보증보험 기본정보 조회|v1|/insurances/basic|POST|기본|
|보증보험-003|보증보험 거래내역 조회|v1|/insurances/transactions|POST|추가|

# 4.2.9 통신 업권

|API ID|API 명|version|URI|HTTP Method|정기적 전송주기|
|---|---|---|---|---|---|
|통신-001|통신 계약 목록 조회|v1|/telecoms|GET|기본|
|통신-002|청구 정보 조회|v1|/telecoms/bills|POST|추가|
|통신-003|통신 거래내역(납입내역) 조회|v1|/telecoms/transactions|POST|추가|
|통신-004|통신 결제내역 조회|v1|/telecoms/paid-transactions|POST|추가|

금융보안원 www.fsec.or.kr

# 금융분야 마이데이터 표준API 규격

# 4.2.10 P2P 업권

|API ID|API 명|version|industry|resource|HTTP Method|정기적 전송주기|
|---|---|---|---|---|---|---|
|P2P-001|P2P 대출 목록 조회|v1| |/lendings|GET|기본|
|P2P-002|P2P 대출 기본정보 조회|v1| |/lendings/basic|POST|기본|
|P2P-003|P2P 대출 추가정보 조회|v1| |/lendings/detail|POST|추가|
|P2P-004|P2P 대출 거래내역 조회|v1| |/lendings/transactions|POST|추가|

# 4.2.11 인수채권 업권

ㅇ 4.2.1의 채권-001~채권-003 API 외 추가 제공해야 할 API 미존재

# 4.2.10 대부 업권

ㅇ 4.2.1의 채권-001~채권-003 API 외 추가 제공해야 할 API 미존재

금융보안원 www.fsec.or.kr

# 금융분야 마이데이터 표준API 규격

# 4.3 지원 API

종합포털이 마이데이터 산업을 지원하기 위해 필요한 API로, 종합포털이 제공하는 API(마이데이터사업자 및 정보제공자가 호출)와 마이데이터사업자 및 정보제공자가 제공하는 API(종합포털이 호출)로 구분 (상세 명세는 제7장 참조)

# 4.3.1 지원 API (종합포털 제공)

|API ID|API 명|version|industry|URI|HTTP Method|
|---|---|---|---|---|---|
|지원-001|접근토큰 발급|해당|없음|/mgmts/oauth/2.0/token|POST|
|지원-002|기관정보 조회|v1|해당|/mgmts/orgs|GET|
|지원-003|서비스정보 조회|v1|해당|/mgmts/services|GET|
|지원-004|마이데이터사업자/정보수신자 통계자료 전송|v1|해당|/mgmts/statistics/mydata|POST|
|지원-005|정보제공자 통계자료 전송|v1|해당|/mgmts/statistics/provider|POST|
|지원-006|통합인증기관용 기관정보 조회|v1|해당|/mgmts/orgs_for_ca|GET|

# 4.3.2 지원 API (마이데이터사업자/정보제공자/중계기관 제공)

|API ID|API 명|version|industry|URI|HTTP Method|
|---|---|---|---|---|---|
|지원-101|접근토큰 발급|해당|없음|/mgmts/oauth/2.0/token|POST|
|지원-102|정보제공자 상태 조회|v1|해당|/mgmts/status|GET|
|지원-103|정보주체 별 전송요구 내역 조회|v1|해당|/mgmts/consents|POST|
|지원-104|통계자료 재전송 요청|v1|해당|/mgmts/req-statistics|GET|

금융보안원 www.fsec.or.kr - 63 -

