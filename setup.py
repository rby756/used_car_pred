import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Car-Price-Prediction-using-MLflow-and-DVC"
AUTHOR_USER_NAME = "rby756"
SRC_REPO = "used_car_pred"
AUTHOR_EMAIL = "torebyv@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="An end to end machine learning project implementation of used car price predcition using mlflow and dvc",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src/carPricePrediction"},
    packages=setuptools.find_packages(where="src")
)