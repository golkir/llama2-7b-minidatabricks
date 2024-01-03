from setuptools import setup, find_packages

setup(
    name='llama-2-7b-minidatabricks',
    packages=find_packages(exclude=[]),
    include_package_data=True,
    version='1.0',
    license='Apache 2.0',
    description='Finetuning Llama2 7B model with Databriks Dolly dataset',
    author='Kirill Goltsman',
    author_email='goltsmank@gmail.com',
    long_description_content_type='text/markdown',
    url='',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformers',
        'llm',
        'finetuning LLMs'
    ],
    install_requires=[
        'accelerate==0.21.0',
        'peft==0.4.0',
        'bitsandbytes==0.40.2',
        'transformers==4.30.0',
        'trl==0.4.7',
        'huggingface_hub',
        'torch==2.1.0',
        'scipy==1.11.4'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Apache 2.0',
        'Programming Language :: Python :: 3.6',
    ],
)
