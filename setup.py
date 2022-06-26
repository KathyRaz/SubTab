from setuptools import setup, find_packages

setup(
    name='explain_ed',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    project_urls={
        'Git': 'https://github.com/KathyRaz/SubTab',
    },
    install_requires=[
        'efficient-apriori',
        'gensim',
        'Jinja2',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
    ]
)
