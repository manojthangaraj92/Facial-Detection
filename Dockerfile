FROM python:3.9.1
ADD . /Facial-Detection
WORKDIR /Facial-Detection
RUN pip install -r requirements.txt
EXPOSE 5000
#CMD ["Facial-Detection/app.py"]
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]