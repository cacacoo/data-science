# coding=utf-8
from collections import Counter

# 'r'는 read-only, 'w'는 writable, 'a'는 append 붙이기
file_for_reading = open('file_name', 'r')

file_for_reading.close()    # 반드시 쓰고 난후 닫을 것


# with을 사용하여 닫아주는 로직을 넣어서 사용할 것~
# with open(filename, 'r') as f
#     data = function_that_gets_data_from(f)
# process(data)


def get_domain(email_address):
    return email_address.lower().split("@")[-1]

with open('email_addresses.txt','r') as f:
    domain_counts = Counter(get_domain(line.strip())
                            for line in f
                            if "@" in line)