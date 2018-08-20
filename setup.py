from distutils.core import setup

from setuptools import find_packages

DISTNAME = 'sfgad'
DESCRIPTION = 'A statistical framework for graph anomaly detection'
MAINTAINER = 'Simon Sudrich'
MAINTAINER_EMAIL = 'uzcyg@student.kit.edu'
URL = 'https://github.com/sudrich/sf-gad'
DOWNLOAD_URL = 'https://api.github.com/repos/sudrich/sf-gad/tarball/master'
LICENSE = 'GPL-3.0'

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

VERSION = '0.1.0'

SCIPY_MIN_VERSION = '0.13.3'
NUMPY_MIN_VERSION = '1.8.2'
PANDAS_MIN_VERSION = '0.22.0'
NETWORKX_MIN_VERSION = '1.11.0'
JOBLIB_MIN_VERSION = '0.12.2'
MYSQL_CONNECTOR_MIN_VERSION = '8.0.12'


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type='text/markdown',
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    version=VERSION,
                    classifiers=[],
                    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
                    install_requires=[
                        'numpy>={0}'.format(NUMPY_MIN_VERSION),
                        'pandas>={0}'.format(PANDAS_MIN_VERSION),
                        'scipy>={0}'.format(SCIPY_MIN_VERSION),
                        'networkx=={0}'.format(NETWORKX_MIN_VERSION),
                        'joblib>={0}'.format(JOBLIB_MIN_VERSION),
                        'mysql-connector-python>={0}'.format(MYSQL_CONNECTOR_MIN_VERSION)
                    ],
                    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
