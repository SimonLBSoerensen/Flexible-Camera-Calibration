import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='fcc',
     version='0.1',
     author="Simon L. B. Sørensen, Oliver H. Vea, Malte N. Jensen and Jakob Yde-Madsen",
     author_email="simso16@student.sdu.dk",
     description="Flexible Camera Calibration",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/SimonLBSoerensen/Flexible-Camera-Calibration",
     packages=setuptools.find_packages(),
     keywords="camera calibration",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     install_requires=["numpy", "opencv-python", "tqdm"],
 )