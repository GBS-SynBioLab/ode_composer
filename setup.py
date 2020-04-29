from setuptools import setup, find_packages

setup(
    name="ode_composer",
    version="0.1",
    description="Building ODE models from data",
    url="https://github.com/zoltuz/ode_composer_py",
    author="Zoltan A Tuza",
    author_email="zoltuz@gmail.com",
    license="MIT",
    packages=find_packages(include=["ode_composer", "ode_composer.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "cvxpy",
        "sklearn",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
)
