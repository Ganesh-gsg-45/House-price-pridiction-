from setuptools import find_packages,setup
from typing import List
hypon_dot='-e.'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if hypon_dot in requirements:
            requirements.remove(hypon_dot)
    return requirements
setup(
    name='houseprice pridiction',
    version='0.0.1',
    author='ganesh',
    author_email='tarigondaganesh1234@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)