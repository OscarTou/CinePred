FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip && \
  pip install -r requirements.txt

COPY CinePred /CinePred

CMD uvicorn CinePred.api.fast:app --host 0.0.0.0 --port $PORT
