from setuptools import setup, find_packages

setup(name='HD_BET',
      version='1.0',
      description='Tool for brain extraction',
      url='https://github.com/MIC-DKFZ/hd-bet',
      python_requires='>=3.5',
      author='Fabian Isensee',
      author_email='f.isensee@dkfz.de',
      license='Apache 2.0',
      zip_safe=False,
      install_requires=[
      'numpy',
      'torch>=0.4.1',
      'scikit-image',
      'SimpleITK'
      ],
      scripts=['HD_BET/hd-bet'],
      packages=find_packages(include=['HD_BET']),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )

