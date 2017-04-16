# coding=utf-8
import re
import sys

# sys.argv[0]은 프로그램 이름, [1]은 커맨드라인에서 주어지는 정규식표현
regex = sys.argv[1]

for line in sys.stdin:
    if re.search(regex, line):
        sys.stdout.write(line)
