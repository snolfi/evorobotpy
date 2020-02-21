# evorobotpy
A tool for training robots in simulation through evolutionary and reiforcement learning methods

All the software required is available ready to be used through the docker container [docker container](https://hub.docker.com/r/vkurenkov/cognitive-robotics) prepared by Vladislav Kurenkov that can be pulled, built and run through the following commands:

```
# Download the container
docker pull vkurenkov/cognitive-robotics

# Build the contaienr
docker build -t vkurenkov/cognitive-robotics .

# Run container
docker run -it \
  -p 6080:6080 \
  -p 8888:8888 \
  --mount source=cognitive-roboitcs-opt-volume,target=/opt \
  cognitive-robotics
```
