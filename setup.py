from setuptools import setup, find_packages

setup(
    name='bert_attn_viz',
    version='0.0.2',
    description="Visualize the self-attention layers in BERT",
    author='Mohd Shukri Hasan',
    author_email='mohdshukri@seekasia.com',
    url='',
    license='Apache 2.0',
    packages=find_packages(exclude=['tests', 'notebooks']),
    # put requirements for documentation purposes (eventhough databricks will ignore it)
    install_requires=['tensorflow>=1.12.0']
)