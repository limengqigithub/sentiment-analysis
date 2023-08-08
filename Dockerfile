FROM python:3.8-slim-buster
COPY . /sentiment-analysis
WORKDIR /sentiment-analysis
RUN pip install -r requirments.txt
EXPOSE 8868
CMD ["python", "app.py"]