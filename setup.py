from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kklgb*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kklgb',
    version='1.0.0',
    description='lightgbm wrapper library.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kklgb",
    author='kazuking',
    author_email='kazukingh01@gmail.com',
    license='Public License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'alembic>=1.6.5',
        'attrs>=21.2.0',
        'cliff>=3.8.0',
        'cmaes>=0.8.2',
        'cmd2>=2.1.2',
        'colorama>=0.4.4',
        'colorlog>=5.0.1',
        'greenlet>=1.1.0',
        'importlib-metadata>=4.6.1',
        'joblib>=1.0.1',
        'lightgbm>=3.2.1',
        'Mako>=1.1.4',
        'MarkupSafe>=2.0.1',
        'numpy>=1.21.1',
        'optuna>=2.8.0',
        'packaging>=21.0',
        'pbr>=5.6.0',
        'prettytable>=2.1.0',
        'pyparsing>=2.4.7',
        'pyperclip>=1.8.2',
        'python-dateutil>=2.8.2',
        'python-editor>=1.0.4',
        'PyYAML>=5.4.1',
        'scikit-learn>=0.24.2',
        'scipy>=1.7.0',
        'six>=1.16.0',
        'SQLAlchemy>=1.4.22',
        'stevedore>=3.3.0',
        'threadpoolctl>=2.2.0',
        'tqdm>=4.61.2',
        'typing-extensions>=3.10.0.0',
        'wcwidth>=0.2.5',
        'zipp>=3.5.0',
    ],
    python_requires='>=3.7'
)
