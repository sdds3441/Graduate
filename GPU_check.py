import tensorflow as tf
from tensorflow.python.client import device_lib

# GPU 장치 목록을 출력
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print(f"총 {len(gpus)}개의 GPU를 발견했습니다:")
    for gpu in gpus:
        print(gpu)
else:
    print("GPU 장치를 찾을 수 없습니다.")

# 세부 정보 확인을 위한 전체 장치 목록 출력
print("\n전체 장치 목록:")
print(device_lib.list_local_devices())
