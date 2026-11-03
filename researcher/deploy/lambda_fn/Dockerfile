# Build from the Hackathon folder:   docker build -f researcher/deploy/lambda_fn/Dockerfile -t atr-run-experiment .
# Contains MAVERIC's radp package, the trained twin and the minimum simulation data (topology + grid).
FROM public.ecr.aws/lambda/python:3.10

COPY researcher/deploy/lambda_fn/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
# a compiler, in case any dependency has no wheel for Amazon Linux 2
RUN yum install -y gcc gcc-c++ && yum clean all
RUN pip install --upgrade pip && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY ["maveric-1.0.0 1/maveric-1.0.0/radp", "${LAMBDA_TASK_ROOT}/radp"]
COPY researcher/atr ${LAMBDA_TASK_ROOT}/atr
COPY researcher/policies ${LAMBDA_TASK_ROOT}/policies
COPY researcher/models/twin_model_map.pickle ${LAMBDA_TASK_ROOT}/models/twin_model_map.pickle
COPY data/sim_data/sim_data/3cell ${LAMBDA_TASK_ROOT}/data/3cell
COPY researcher/deploy/lambda_fn/handler.py researcher/deploy/lambda_fn/protocol_handler.py \
     researcher/deploy/lambda_fn/lab_handler.py researcher/deploy/lambda_fn/console_api.py ${LAMBDA_TASK_ROOT}/

ENV MAVERIC_ROOT=/var/task ATR_SIM_DATA=/var/task/data/3cell ATR_RUNS_DIR=/tmp/runs MPLCONFIGDIR=/tmp/mpl
# one image, four functions: run_experiment (default), protocol tasks, lab executor, console API (the CDK stack sets CMD)
CMD ["handler.handler"]
