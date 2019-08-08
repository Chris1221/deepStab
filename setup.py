fom distutils.core import setup

setup(
        name='deepstab',
        version='0.1a',
        packages=['deepstab'],
        install_requires=[
            'numpy',
            'matplotlib',
            'pysam',
            'click',
            'gffutils',
            'pyfaidx',
            'tqdm'
        ]
)
