FROM python:3.11

WORKDIR /app
COPY src src
COPY pyproject.toml .
RUN pip install .

COPY models .

CMD ["fastapi", "run", "src/afterlife/cmds/server.py", "--port", "80"]




