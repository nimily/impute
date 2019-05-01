from setuptools import setup

setup(name='lr-impute',
      version='0.1',
      description='Implementation of low-rank imputing methods for general sampling distributions',
      url='git@github.com:nimily/low-rank-impute.git',
      author='Nima Hamidi',
      author_email='nimaa.hamidi@gmail.com',
      license='MIT',
      packages=['impute'],
      install_requires=['numpy >= 1.16.2',
                        'scikit-learn >= 0.20.3'],
      zip_safe=False)
