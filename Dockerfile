FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# CMD ["python", "main.py", "start-grid", "--ticker", "BTC/KRW", "--lower-price", "30000000", "--upper-price", "40000000", "--grid-count", "5", "--order-amount-krw", "10000"]
# 위 CMD 명령어는 예시입니다. 실제 실행 시에는 환경 변수 등을 활용하여 유연하게 설정하는 것을 권장합니다.
# 예를 들어, Docker run 명령 시 -e 옵션으로 환경 변수를 전달하거나,
# main.py에서 argparse 등을 사용하여 인자를 받을 수 있습니다.
# 현재는 main.py에 인자를 받는 로직이 없으므로, 주석 처리합니다.
# 실제 사용 시 main.py를 수정하여 인자를 받도록 구현해야 합니다.
CMD ["python", "main.py"]