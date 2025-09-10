FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

WORKDIR /code/app

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1