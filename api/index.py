import sys
import os

# 부모 디렉토리(루트)를 sys.path에 추가하여 main.py를 안전하게 불러옵니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import app
