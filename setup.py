from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="py_backtester_suite",
    version="1.0.0",
    author="Kunal Chhonkar",
    author_email="chhonkar002ai@gmail.com",
    description="A comprehensive event-driven backtester for financial trading strategies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    keywords="backtesting trading finance quantitative-finance algorithmic-trading",
)