from setuptools import setup, find_packages

setup(
    name='esis',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'sphinx-autodoc-typehints',
        'astropy-sphinx-theme',
        'kgpy @ git+https://github.com/Kankelborg-Group/kgpy.git#egg=kgpy',
        'ndcube',
        'tensorflow',
        'pylatex',
        'num2words'
    ],
    url='https://github.com/Kankelborg-Group/ESIS',
    license='',
    author='Roy Smart',
    author_email='roytsmart@gmail.com',
    description='Repository for the EUV Snapshot Imaging Spectrograph (ESIS).'
)
