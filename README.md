# Upbit 고빈도 퀀트 스캘핑 봇

## 📖 개요

이 프로젝트는 업비트(Upbit) 거래소의 1분봉 데이터를 기반으로 하는 고빈도 퀀트 스캘핑(Scalping) 자동매매 봇입니다. XGBoost 머신러닝 모델을 사용하여 초단기 가격 변동을 예측하고, 이를 기반으로 거래를 실행합니다. 또한, 강화학습(RL) 에이전트를 훈련하고 배포하는 파이프라인을 포함하고 있습니다.

## ✨ 주요 기능

- **데이터 수집:** `ccxt`를 사용하여 업비트에서 지정된 기간과 티커의 1분봉 OHLCV 데이터를 다운로드합니다.
- **데이터 전처리:** 다운로드한 데이터를 기반으로 RSI, MACD 등 다양한 기술적 분석(TA) 지표를 생성합니다.
- **모델 훈련:** 전처리된 데이터를 사용하여 가격 상승/하락을 예측하는 XGBoost 분류 모델을 훈련합니다.
- **백테스팅:** 훈련된 모델을 사용하여 과거 데이터에 대한 거래 시뮬레이션을 실행하고 성과를 평가합니다.
- **실시간 거래:** 실시간으로 데이터를 받아 모델의 예측에 따라 자동으로 매매를 실행합니다.
- **CI/CD:** GitHub Actions를 통해 코드 품질 검사, Docker 이미지 빌드, AWS EC2 배포까지의 과정이 자동화되어 있습니다.

## 📂 프로젝트 구조

```
/
├── .github/workflows/deploy.yml  # CI/CD 파이프라인
├── .gitignore
├── Dockerfile                    # Docker 이미지 생성을 위한 설정 파일
├── advanced_backtester.py        # 백테스팅 엔진
├── ccxt_downloader.py            # 데이터 다운로더
├── data/                         # 원본 OHLCV 데이터 (CSV)
├── cache/                        # 전처리된 데이터 (Feather)
├── live_trader.py                # 실시간 거래 엔진
├── main.py                       # 프로그램의 메인 실행 파일
├── model_trainer.py              # XGBoost 모델 훈련
├── preprocessor.py               # 데이터 전처리기
├── price_predictor.pkl           # 훈련된 XGBoost 모델
├── price_scaler.pkl              # 데이터 스케일러
└── requirements.txt              # Python 의존성 목록
```

## 🚀 시작하기

### 사전 준비

- Python 3.12 이상

### 설치

1.  저장소를 클론합니다.
    ```bash
    git clone https://github.com/AzureMooni/upbit_auto_bot.git
    cd upbit_auto_bot
    ```

2.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

3.  `pandas-ta` 라이브러리를 소스에서 직접 설치합니다. (워크플로우 기준)
    ```bash
    pip install git+https://github.com/aarigs/pandas-ta.git
    ```

## 🛠️ 사용법

`main.py`를 사용하여 다양한 모드를 실행할 수 있습니다.

- **데이터 다운로드:**
  ```bash
  python main.py --mode download --start-date 2025-09-01 --end-date 2025-10-01
  ```

- **데이터 전처리:**
  ```bash
  python main.py --mode preprocess
  ```

- **모델 훈련:**
  ```bash
  python main.py --mode train
  ```

- **백테스트 실행:**
  ```bash
  python main.py --mode backtest --start-date 2025-09-01 --end-date 2025-10-01 --capital 50000
  ```

- **실시간 거래 시작:**
  ```bash
  python main.py --mode trade --capital 100000
  ```

## ⚙️ CI/CD 파이프라인

이 프로젝트는 GitHub Actions를 사용하여 CI/CD 파이프라인을 구축했습니다.

- **Pull Request:** `main` 브랜치로 Pull Request를 생성하면, `ruff`를 사용한 린트 검사가 자동으로 실행됩니다.
- **Push to Main:** `main` 브랜치에 코드가 푸시되면, 린트 검사 후 Docker 이미지를 빌드하여 Docker Hub에 푸시하고, 최종적으로 AWS EC2에 애플리케이션을 배포합니다.
- **Scheduled Run:** 매주 일요일 자정에 데이터 수집, 전처리, 모델 훈련 및 배포가 자동으로 실행됩니다.
