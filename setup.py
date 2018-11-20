from setuptools import setup

setup(name='HD_BET',
      version='1.0',
      description='Tool for brain extraction',
      url='',  # TODO,
      python_requires='>=3.5',
      author='Fabian Isensee',
      author_email='f.isensee@dkfz.de',
      license='MIT',  # TODO
      zip_safe=False,
      install_requires=[
      'numpy',
      'torch>=0.4.1',
      'scikit-image',
      'SimpleITK'
      ],
      scripts=['HD_BET/hd-bet'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )

