FROM python:3.7.10-slim-buster

RUN export DEBIAN_FRONTEND=noninteractive \
  && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
  && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
  && apt update && apt install -y locales \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
  
RUN pip install \
  torch==1.12.1+cpu \
  torchvision==0.13.1+cpu \
  -f https://download.pytorch.org/whl/cpu/torch_stable.html \
  && rm -rf /root/.cache/pip


RUN mkdir -p -v /home/app



ADD requirements.txt /home/app



RUN pip install --no-cache-dir -r /home/app/requirements.txt

ADD /src/demo_file.py /home/app

RUN mkdir -p -v /home/app/src/utils
ADD /src/utils /home/app/src/utils

EXPOSE 8080

WORKDIR /home/app/

RUN touch setup.py
ENTRYPOINT [ "python","demo_file.py"]


# ENTRYPOINT []