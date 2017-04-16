# coding=utf-8
import sys
from collections import Counter

# 출력하고 싶은 단어의 수를 첫 번째 인자로 입력
try:
    num_words = int(sys.argv[1])
except:
    print "usage: most_common_words.py num_words"
    sys.exit(1)

counter = Counter(word.lower()  # 소문자 변환
                  for line in sys.stdin
                  for word in line.strip().split()  # 띄워쓰기 기준 나누기
                  if word)  # 빈문자열 제거

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write(" : ")
    sys.stdout.write(word)
    sys.stdout.write("\n")
