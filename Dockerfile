FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install grpcio grpcio-tools protobuf

# Copy files
COPY . .

# Generate proto files
RUN python -m grpc_tools.protoc -I./protos --python_out=./tetra --grpc_python_out=./tetra ./protos/remote_execution.proto

WORKDIR /app/src

EXPOSE 50051

CMD ["python", "server.py"]