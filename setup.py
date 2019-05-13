from setuptools import findpackages, setup

DISTNAME = "math_implementations"
AUTHOR = "camen"
DESCRIPTION = "Implementations of various maths"


def get_version():
    """Read version number from VERSION file."""
    with open("VERSION") as buff:
        return buff.read()


def get_long_description():
    """Read long description from README.md file."""
    with open("README.md") as buff:
        return buff.read()


def get_dev_requirements():
    """Read testing and development requirements from requirements-dev.txt."""
    with open("requirements-dev.txt") as buff:
        return buff.readlines()


def get_install_requirements():
    """Read installation requirements from requirements.txt."""
    with open("requirements.txt") as buff:
        return buff.readlines()


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=get_version(),
        author=AUTHOR,
        packages=find_packages(),
        include_package_data=True,
        python_requires=">=3.6",
        install_requires=get_install_requirements(),
        tests_require=get_dev_requirements(),
        description=DESCRIPTION,
        long_description=get_long_description
    )
