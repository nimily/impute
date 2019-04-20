from setuptools import setup

setup(name='soft-impute',
      version='0.1',
      description='Implementation of soft-impute for general sampling distributions',
      url='git@github.com:nimily/general-soft-impute.git',
      author='Nima Hamidi',
      author_email='nimaa.hamidi@gmail.com',
      license='MIT',
      packages=['impute'],
      install_requires=['numpy >= 1.16.2',
                        'sklearn >= 0.20.3'],
      zip_safe=False)
