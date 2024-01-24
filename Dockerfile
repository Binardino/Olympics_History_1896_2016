FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y 
RUN pip install --upgrade pip && pip install -r requirements.txt

#Expose port 8501
EXPOSE 8501

#set environment variables for streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/app_olympics.py", "--server.port=8501", "--server.address=0.0.0.0"]