from setuptools import setup, find_packages

install_requires=[
    "transformers@git+https://github.com/huggingface/transformers@main",
]
setup(
    name = 'NLP4Code',
    packages = find_packages(),
    install_requires=install_requires,
)