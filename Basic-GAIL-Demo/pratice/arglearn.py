import argparse

parser = argparse.ArgumentParser(description='命令行中传入一个数字')  #명령줄에 숫자 전달
# help는 매개변수의 프롬프트 정보
parser.add_argument('integers', type=str, nargs='+', help='传入的数字')  # 수신번호

args = parser.parse_args()

print(args.integers)