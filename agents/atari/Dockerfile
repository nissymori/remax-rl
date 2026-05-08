# envpool it's not compatible with newer versions of jax
FROM nvcr.io/nvidia/jax:24.04-py3

# install 
WORKDIR /app/
COPY . .

RUN pip install -e .[atari] 

