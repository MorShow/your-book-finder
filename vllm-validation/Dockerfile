FROM nvidia/cuda:12.5.0-base-ubuntu22.04

RUN apt update && apt install -y \
    python3.10 python3-pip git curl libaio-dev \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && pip install --upgrade pip
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu125
RUN pip install flash-attn==2.5.5 --no-build-isolation
RUN pip install vllm==0.8.5

CMD ["python"]