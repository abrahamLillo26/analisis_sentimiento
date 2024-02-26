FROM python:3.8.10-slim-buster

# RUN apt update \
# 	&& apt install -y curl \
# 	&& rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt-get -y upgrade
RUN apt-get -y update

WORKDIR /opt/alloxentric

COPY ./requirements.txt /opt/alloxentric/.

# Agregar cuando corresponda
RUN pip install -r requirements.txt
# RUN apt-get -y update
# RUN apt-get -y upgrade
# RUN apt-get install -y ffmpeg

# Copia recursiva del contenido del directorio actual dentro del contenedor
COPY . /opt/alloxentric

CMD ["python3", "analisis_sentimiento.py"]
