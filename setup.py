#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README_clip_benchmark.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

def load_requirements(f):
    return [l.strip() for l in open(f).readlines()]

requirements = load_requirements("requirements.txt")

test_requirements = requirements + ["pytest", "pytest-runner"]

setup(
    author="Laure Ciernik",
    author_email='your.email@example.com',  # Replace with your email
    python_requires='>=3.9',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description=(
        "A package for analyzing consistency of representational similarities and "
        "evaluating vision foundation models with linear probes. "
        "Based on CLIP-benchmark (https://github.com/LAION-AI/CLIP_benchmark) by Mehdi Cherti."
    ),
    entry_points={
        'console_scripts': [
            'sim_consistency=sim_consistency.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='sim_consistency, representation learning, vision models, similarity analysis',
    name='sim_consistency',
    packages=find_packages(include=['sim_consistency', 'sim_consistency.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/lciernik/similarity_consistency',
    version='0.1.0',
    zip_safe=False,
    extras_require={
        "vtab": ["task_adaptation==0.1", "timm>=0.5.4"],
        "tfds": ["tfds-nightly", "timm>=0.5.4"],
        "coco": ["pycocotools>=2.0.4"],
    }
)