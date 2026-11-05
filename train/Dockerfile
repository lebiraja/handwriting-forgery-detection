# Reuses the local laya-laya image (torch 2.6.0+cu124, torchvision, PIL, CUDA verified on the RTX 4050).
# Only the small missing packages are added; torch is not reinstalled.
FROM laya-laya:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -c "import torch;print(torch.__version__)"
COPY src ./src
ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
CMD ["python", "-m", "src.train", "--config", "src/config.yaml"]
