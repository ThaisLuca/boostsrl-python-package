# Copyright (c) 2017-2018 StARLinG Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# See <http://www.gnu.org/licenses/> or the license in the base of
# the repository at <https://github.com/starling-lab/boostsrl-python-package>

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='boostsrl',
    packages=['boostsrl'],
    author='Alexander L. Hayes (batflyer)',
    author_email='alexander@batflyer.net',
    version='1.0.1',
    description='Python wrappers for using BoostSRL jar files.',

    # boostsrl_data stores files in the user's home directory.
    include_package_data = True,
    data_files=[(path.expanduser('~') + '/.boostsrl_data', ['boostsrl/v1-0.jar',
                                                            'boostsrl/auc.jar']),
                (path.expanduser('~') + '/.boostsrl_data/train', ['boostsrl/train/train_bk.txt']),
                (path.expanduser('~') + '/.boostsrl_data/test', ['boostsrl/test/test_bk.txt'])],

    # Project's main homepage.
    url='https://github.com/starling-lab/boostsrl-python-package',
    download_url="https://github.com/starling-lab/boostsrl-python-package/archive/v1.0.1.tar.gz",

    # License
    license='GPL-3.0',

    classifiers=[
        # Current development status
        'Development Status :: 3 - Alpha',

        # Intended Audiences
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        # License
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # OS
        'Operating System :: POSIX :: Linux',

        # Supported Python Versions
        # Check build status: https://travis-ci.org/starling-lab/boostsrl-python-package
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        # Topic
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],

    # Relevant keywords (from boost-starai/BoostSRL)
    keywords='machine-learning-algorithms machine-learning statistical-learning pattern-classification artificial-intelligence',

    install_requires = [],
    extras_require={
        'test': ['coverage']
    }

)
