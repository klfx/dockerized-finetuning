FROM python:3.11.6

WORKDIR /finetune-workdir

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY finetune.py .

ENTRYPOINT ["python","./finetune.py"]