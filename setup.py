from setuptools import setup, find_packages

setup(
    name="opencv-mariokartwii",  # Package name (should match GitHub repo)
    version="0.1.0",  # Initial version
    author="Brayden Gess",  # Your name
    description="An OpenCV-based project for Mario Kart Wii analysis",
    url="https://github.com/BraydenGess/opencv-mariokartwii",
    packages=find_packages(),  # Automatically finds packages in the repo
    install_requires=[
        "opencv-python",
        "torch",
        "pygame",
        "numpy"
    ],  # Add other dependencies here
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version required
)