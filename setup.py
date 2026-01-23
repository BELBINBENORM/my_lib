from setuptools import setup

setup(
    name="select_regression",
    version="0.1",
    py_modules=["select_regression"], 
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib',
        'psutil',      
        'matplotlib',  
    ],
)
