from setuptools import setup, find_packages

setup(
        name='auctionbid',
        version="0.0.1",
        packages=find_packages(),
        platforms=["all"],
        install_requires=[
            "numpy",
            "pandas",
            # "torch==1.12.0",
            "gin",
            "gin_config",
            "matplotlib",
            "scipy",
            "psutil",
            "func_timeout",
        ]
    )
