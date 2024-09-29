# 첨부12

|구분|분류|API 구분|코드|API 명 (resource만 표기)|
|---|---|---|---|---|
|개별인증|인증|AU01|/oauth/2.0/authorize| |
| |AU02|/oauth/2.0/token (접근토큰 발급 요청)| | |
| |AU03|/oauth/2.0/token (접근토큰 갱신)| | |
| |AU04|/oauth/2.0/revoke| | |
|통합인증|AU11|/oauth/2.0/token| | |
|공통|CM01|/apis| | |
| | |CM02|/consents| |
|계좌 목록|BA01|/accounts| | |
| | |BA02|/accounts/deposit/basic| |
| | |BA03|/accounts/deposit/detail| |
| | |BA04|/accounts/deposit/transactions| |
| | |BA11|/accounts/invest/basic| |
| | |BA12|/accounts/invest/detail| |
| | |BA13|/accounts/invest/transactions| |
|대출상품 정보|BA21|/accounts/loan/basic| | |
| | |BA22|/accounts/loan/detail| |
| | |BA23|/accounts/loan/transactions| |
|카드 목록|CD01|/cards| | |
| | |CD02|/cards/{card_id}| |
|카드 정보|CD03|/cards/{card_id}/approval-domestic| | |
| | |CD04|/cards/{card_id}/approval-overseas| |
|포인트 정보|CD11|/points| | |
|청구 및 결제 정보|CD21|/bills| | |
| | |CD22|/bills/detail| |
| | |CD23|/payments| |
| | |CD24|/payments/revolving| |
| | |CD31|/loans| |
|대출상품 정보|CD32|/loans/short-term| | |
| | |CD33|/loans/long-term| |
|계좌 목록|IV01|/accounts| | |
| | |IV02|/accounts/basic| |
| | |IV03|/accounts/transactions| |

금융보안원 www.fsec.or.kr - 295 -

# 금융분야 마이데이터

# 표준API 규격

# 보험업권

|목록|정보|
|---|---|
|IS01|/insurances|
|IS02|/insurances/basic|
|IS03|/insurances/contracts|
|IS04|/insurances/car|
|IS05|/insurances/payment|
|IS06|/insurances/transactions|
|IS07|/insurances/car/transactions|
|IS08|/insurances/coverages|

# 대출업권

|목록|정보|
|---|---|
|IS11|/loans|
|IS12|/loans/basic|
|IS13|/loans/detail|
|IS14|/loans/transactions|

# 선불전자지급수단

|목록|정보|
|---|---|
|EF01|/prepaid|
|EF02|/prepaid/balance|
|EF03|/prepaid/charge|
|EF04|/prepaid/transactions|

# 전자금융업권

|계정 목록|결제정보|
|---|---|
|EF11|/paid|
|EF12|/paid/methods|
|EF13|/paid/transactions|

# 할부금융업권

|대출상품 정보| | | | |
|---|---|---|---|---|
| | | |CP01|/loans|
| | | |CP02|/loans/basic|
| | | |CP03|/loans/detail|
| | | |CP04|/loans/transactions|
| | | |CP05|/loans/oplease/basic|
| | | |CP06|/loans/oplease/transactions|

# 보증보험업권

|보증보험 목록|정보|
|---|---|
|GI01|/insurances|
|GI02|/insurances/basic|
|GI03|/insurances/transactions|

# 통신업권

|통신 목록|정보|
|---|---|
|TC01|/telecoms|
|TC02|/telecoms/bills|
|TC03|/telecoms/transactions|
|TC04|/telecoms/paid-transactions|

금융보안원 www.fsec.or.kr - 296 -

# 금융분야 마이데이터 표준API 규격

# 대출 목록

# P2P 업권 대출 정보

# 개인형 IRP 목록

# 은행, 보험, 금투업권 개인형 IRP 정보

# 선불카드 목록

# 은행, 카드업권 선불카드 정보

# 인수채권/금전대부 목록

# 인수채권, 대부업권 인수채권/금전대부 정보

|API 코드|API 경로|
|---|---|
|LD01|/lendings|
|LD02|/lendings/basic|
|LD03|/lendings/detail|
|LD04|/lendings/transactions|
|IR01|/irps|
|IR02|/irps/basic|
|IR03|/irps/detail|
|IR04|/irps/transactions|
|PP01|/prepaid|
|PP02|/prepaid/balance|
|PP03|/prepaid/transactions|
|PP04|/prepaid/approval|
|BD01|/bonds|
|BD02|/bonds/detail|
|BD03|/bonds/transactions|
