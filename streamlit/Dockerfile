FROM python:3.12-slim

WORKDIR /code/app

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE ${PORT}

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health

CMD exec streamlit run main.py --server.port ${PORT} --server.address=0.0.0.0

# ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]