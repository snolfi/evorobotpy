# evorobotpy
A tool for training robots in simulation through evolutionary and reinforcement learning methods

It includes a [guide](/doc/LearningHowToTrainRobots.pdf) that gently introduce the topic and explain how to use this and other available tools to train robots in simulation.

All the software required is available ready to be used through the docker container [docker container](https://hub.docker.com/r/vkurenkov/cognitive-robotics) prepared by Vladislav Kurenkov that can be pulled, built and run through the following commands:

```
# Download the container (CPU version)
docker pull vkurenkov/cognitive-robotics:cpu

# Run container (CPU version)
docker run -it \
  -p 6080:6080 \
  -p 8888:8888 \
  --mount source=cognitive-robotics-opt-volume,target=/opt \
  vkurenkov/cognitive-robotics:cpu
  
# Download the container (GPU version)
docker pull vkurenkov/cognitive-robotics:gpu

# Run container (GPU version)
docker run --gpus all -it \
-p 6080:6080 \
-p 8888:8888 \
--mount source=cognitive-robotics-opt-volume,target=/opt \
vkurenkov/cognitive-robotics:gpu
```
Render environments using NoVNC (desktop access via browser at localhost:6080). Code editing using VSCode (you can attach to the container using VSCode and edit the source code conveniently -- allows to use IntelliSense and more). You can use Jupyter Notebook, just run jupyter notebook --ip=0.0.0.0 --port=8888 inside a container and you can access it in your browser at localhost:8888. The changes made to the source code are persistent (e.g. you can restart the container and your changes won't be lost)

As an alternative to the docker container, you should clone evorobotpy and install:
1) Python 3.5+ and the cython, pyglet and matplotlib packages
2) [AI Gym](gym.openai.com)
3) [GNU Scientific Library](https://www.gnu.org/software/gsl)
4) [Pybullet](https://pybullet.org/)
5) [Baselines](https://github.com/openai/baselines) (optional)
6) [Spinningup](https://spinningup.openai.com/) (optional)

