#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="bee-ai-fact-consolidator",
    version="0.5.0",
    description="A tool to consolidate and deduplicate facts from Bee AI using clustering and a local LLM",
    author="Ian Matson",
    author_email="howdy@ian-matson.com",
    url="https://github.com/imatson9119/bee-ai-fact-consolidator",
    py_modules=["fact_consolidator", "text_similarity"],
    install_requires=[
        "requests",
        "scikit-learn",
        "python-dotenv>=1.0.0",
        "openai",
        "numpy",
        "tqdm",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "bee-fact-consolidator=fact_consolidator:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
) 