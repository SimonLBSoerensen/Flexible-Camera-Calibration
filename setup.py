import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fcc-pkg-SimonLBS",
    version="0.0.1",
    author="Simon L. B. Sørensen and contributors",
    author_email="simonlyckbjaert@hotmail.com",
    description="Flexible Camera Calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SimonLBSoerensen/Flexible-Camera-Calibration",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="camera calibration",
    python_requires='>=3.6',
     install_requires=["numpy", "opencv-python", "tqdm"],
)
