import setuptools
from setuptools import setup


setup(
    name='ml-tdd-tutorial',
    version='0.1',
    description="Test Tutorial For ML/DL",
    # 현재 폴더에서 해당 하는 모든 패키지를 리스트로 리턴
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    # MANIFEST.in을 찾고 패키지데이터와 일치하는 모든 항목을 설치
    # 예를 들어 정적 파일
    include_package_data=True,
    # 패키징을 위한 라이브러리 설치
    install_requires=[
        "attrs==19.3.0",
    ],
)
