# Use the official Triton Inference Server image
FROM nvcr.io/nvidia/tritonserver:23.09-py3

# Copy the model repository to the container
COPY models/ /models/

# Expose the necessary ports
EXPOSE 8000 8001 8002

# Start the Triton Inference Server
CMD ["tritonserver", "--model-repository=/models"]