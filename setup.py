import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CurML",
    version="0.0.1",
    author="THUMNLab/clteam",
    author_email="zhouyw-21@mails.tsinghua.edu.cn",
    description="Curriculum Machine Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/THUMNLab/CurML",
    project_urls={
        "Bug Tracker": "https://github.com/THUMNLab/CurML",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License"
    ],
    packages=setuptools.find_packages(exclude=("tests", "examples", "docs")),
    python_requires=">=3.6",
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'torchvision',
    ],
)