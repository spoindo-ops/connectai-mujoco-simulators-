from setuptools import setup, find_packages

setup(
    name="ball_nav_env",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["gymnasium", "mujoco"],
)
