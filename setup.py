import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flexiblecc",
    version="0.0.10",
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
    license='MIT',
    install_requires=["numpy", "opencv-contrib-python>=3.1", "tqdm", "matplotlib", "scipy", Pillow],
)

#python setup.py sdist bdist_wheel
#python -m twine upload dist/*

