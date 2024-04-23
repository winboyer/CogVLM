import re
# import jieba
# import jieba.posseg as pseg
import spacy
nlp = spacy.load("en_core_web_sm")
nlp_zh = spacy.load("zh_core_web_sm")

import time

# data = 'describe the image'
data = '请描述这张图片'
data = 'There is no bus visible in the image'
# data = 'whether the person is smoking'
# data = 'find a bus in the image'
# data = '找出穿卫衣的男人'
# data = '描述穿卫衣的男人'

# pattern = r"[person|人]"
pattern = r"(.*)[person|人](.*)"

chinese_flag = False
for _char in data:
    if '\u4e00' <= _char <= '\u9fa5':
        print('this is chinese sentence')
        chinese_flag = True
        break

if "describe" in data or "描述" in data:
    print(data)

# result = re.findall(pattern, data)
# print(result)

# tokens = pseg.cut(data)
# total_words = []
# for word, flag in tokens:
#     print('%s %s' % (word, flag))
#     if word not in(' ', ',', '.'):
#         total_words.append(word)
# print(total_words)


t1 = time.time()
if chinese_flag:
    doc = nlp_zh(data)
else:
    doc = nlp(data)
print('time======', (time.time()-t1)*1000)
print(len(doc), doc)

for chunk in doc:
    print(chunk.text, chunk.pos_)

