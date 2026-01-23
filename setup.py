from setuptools import setup

setup(
    name="file_select_regression",
    version="0.1",
    py_modules=["file_select_regression"], # This points to your .py filename
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib',
    ],
)
