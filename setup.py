from setuptools import setup

setup(
    name = "openalpha",
    version = "0.0.2",
    description = "Utilities for OpenAlpha",
    url = "https://github.com/june-nahmgoong/openalpha",
    author = "June Nahmgoong",
    author_email = "june.nahmgoong@openalpha.net",
    packages = ["openalpha"],
    install_requires = ["numpy","pandas","google-cloud-storage"] 
)