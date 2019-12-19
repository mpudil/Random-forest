from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pandas", "numpy", "pytest"]

setup(
    name="sqlrandforest",
    version="0.0.1",
    author="Mitch Pudil",
    author_email="mpudil@andrew.cmu.edu",
    description="A package to create random forest in Python and decision trees in SQL",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/your_package/homepage/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
