# streamlit_vis/setup.py

from setuptools import setup, find_packages

setup(
    name="streamlit-vis-network",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "streamlit_vis": ["build/**", "build/static/**/*"]
    },
    install_requires=[
        "streamlit>=0.63",
    ],
    python_requires=">=3.6",
)