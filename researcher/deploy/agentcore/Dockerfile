# AgentCore Runtime image for the agents. No torch or MAVERIC: experiments run behind the Gateway in the Lambda.
# Build from the Hackathon folder:   docker build -f researcher/deploy/agentcore/Dockerfile -t atr-agents .
# AgentCore Runtime runs ARM64 containers: build with --platform linux/arm64. Base from ECR Public (no Docker Hub limits).
FROM public.ecr.aws/docker/library/python:3.10-slim
WORKDIR /app
COPY researcher/deploy/agentcore/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY researcher/atr atr
COPY researcher/policies policies
COPY researcher/deploy/agentcore_app.py deploy/agentcore_app.py
COPY data/sim_data/sim_data/3cell data/3cell
ENV ATR_MODE=aws ATR_SIM_DATA=/app/data/3cell ATR_RUNS_DIR=/tmp/runs MPLCONFIGDIR=/tmp/mpl PYTHONPATH=/app
EXPOSE 8080
CMD ["python", "deploy/agentcore_app.py"]
