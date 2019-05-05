from setuptools import setup, find_packages

VERSION = '0.1'


def write_version(filename='impute/version.py'):
    with open(filename, 'w') as f:
        print(f"version = '{VERSION}'", file=f)


def setup_package():
    write_version()

    setup(name='impute',
          version=VERSION,
          description='Implementation of low-rank imputing methods for general sampling distributions',
          url='git@github.com:nimily/impute.git',
          author='Nima Hamidi',
          author_email='nimaa.hamidi@gmail.com',
          license='MIT',
          packages=find_packages(),
          install_requires=['numpy >= 1.16.2',
                            'scikit-learn >= 0.20.3',
                            'sklearn >= 0.0'],
          zip_safe=False)


if __name__ == '__main__':
    setup_package()
