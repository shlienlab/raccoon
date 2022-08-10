from setuptools import setup, find_packages
from codecs import open
from os import path

here    = path.abspath(path.dirname(__file__))
version = open("raccoon/_version.py").readlines()[-1].split()[-1].strip("\"'")

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='raccoon-cluster',

    version=version,

    description='Scale-adaptive clustering in Python',
    long_description=long_description,

    url='https://github.com/shlienlab/raccoon',
	download_url = 'https://github.com/shlienlab/raccoon/archive/'+version+'.tar.gz', 
    author='Federico Comitani',
    author_email='federico.comitani@gmail.com',

    license='GPL-3.0',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
	'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
	'Programming Language :: Python :: 3.7',
	'Programming Language :: Python :: 3.8'	
		],

    keywords=['clustering','optimization','dimensionality-reduction',
              'differential-evolution','knn','umap','hierarchical-clustering',
	      'multi-scale','scale-adaptive','optimal-clusters'], 

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['numpy>=1.19.5',
		'pandas>=1.1.3',
		'scikit-learn>=0.22.2.post1',
		'scikit-network==0.20.0',
                'numba>=0.52.0',
		'umap-learn>=0.3.9',
		'optuna>=2.10.0',
		'psutil>=5.7.3',
		'anytree>=2.8.0',
		'matplotlib>=3.3.3',
		'seaborn>=0.11.0'],

    extras_require={ 'gpu': ['cupy==8.60',
				'cuml==0.18',
				'cudf==0.18',
				'cugraph==0.18'],
		     'hdbscan': ['hdbscan']},
    zip_safe=False,

)

