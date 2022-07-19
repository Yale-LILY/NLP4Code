from setuptools import setup, find_packages

install_requires=[
    "transformers@git+https://github.com/huggingface/transformers@29fd471556443e63eda1ee348c43ec6de5ef158c",
]
setup(
    name = 'NLP4Code',
    packages = find_packages(),
    install_requires=install_requires,
)