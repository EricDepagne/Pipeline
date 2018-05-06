from setuptools import setup, find_packages

setup(
    name='pipeline',
    version=__import__('pipeline').__version__,
    description=__import__('pipeline').__doc__,
    long_description=open('README.rst').read(),
    author='Eric Depagne',
    author_email='eric@saao.ac.za',
    url='https://github.com/EricDepagne/pipeline',
    license='BSD 3',
    packages=find_packages(exclude=['tests*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        'numpy',
        'astropy',
        'scipy',
    ],
    include_package_data=True,
    zip_safe=False
)