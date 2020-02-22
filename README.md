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
Render environments using NoVNC (desktop access via browser at localhost:6080). Code editing using VSCode (you can attach to the container using VSCode and edit the source code conveniently -- allows to use IntelliSense and more). You can use Jupyter Notebook, just run jupyter notebook --ip=0.0.0.0 --port=8888 inside a container and you can access it in your browser at localhost:8888. The changes made to the source code are persistent (e.g. you can restart the container and your changes won't be lost)

As an alternative to the docker container, you should clone evorobotpy and install:
1) Python 3.5+ and the cython, pyglet and matplotlib packages
2) [AI Gym](gym.openai.com)
3) [GNU Scientific Library](https://www.gnu.org/software/gsl)
4) [Pybullet](https://pybullet.org/)
5) [Baselines](https://github.com/openai/baselines) (optional)
6) [Spinningup](https://spinningup.openai.com/) (optional)

