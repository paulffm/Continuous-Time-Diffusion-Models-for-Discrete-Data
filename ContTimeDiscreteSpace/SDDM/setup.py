"""Setup."""

from setuptools import setup


setup(name='sddm',
      py_modules=['sddm'],
      install_requires=[
          'ml_collections',
          'numpy',
          'matplotlib',
          'tqdm',
      ]
)
