FROM python:3.11-slim

#RUN apt-get update && apt-get install -y libgomp1

COPY ./app ./app
WORKDIR /app
COPY ./requirements.txt .

COPY ./models /models

RUN pip3 install -r requirements.txt

EXPOSE ${APP_PORT}

VOLUME /models

CMD uvicorn app:app --reload --port ${APP_PORT} --host 0.0.0.0
