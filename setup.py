from setuptools import setup, find_packages

setup(
    name='vsl-recognition',
    version='0.1.0',
    author='Your Name',
    description='Real-time Vietnamese Sign Language Recognition',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'mediapipe>=0.10.9',
        'opencv-python>=4.8.1',
        'tensorflow>=2.15.0',
        'numpy>=1.24.3',
        'scikit-learn>=1.3.2',
    ],
)
