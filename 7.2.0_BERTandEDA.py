from transformers import *
import re
import numpy as np
import pandas as pd
import os

# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', clean_up_tokenization_spaces=False)

def bert_tokenizer(sent, MAX_LEN):

    encoded_dict = tokenizer.encode_plus(
        text = sent1,       # 첫번째 문장
        text_pair = sent2,  # 두번째 문장
        add_special_tokens= True, # 스페셜 토큰 추가
        max_length = MAX_LEN,
        pad_to_max_length = True,
        return_attention_mask = True,
        truncation = True       # MAX_LEN보다 긴 부분을 잘라낸다.
    )

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id

# Tokeninzer checking
# # Special Tokens
# print(tokenizer.all_special_tokens, "\n", tokenizer.all_special_ids)
#
# # Test Tokenizer
# kor_encode = tokenizer.encode("안녕하세요, 반갑습니다")
# eng_encode = tokenizer.encode("Hello world")
# kor_decode = tokenizer.decode(kor_encode)
# eng_decode = tokenizer.decode(eng_encode)
#
# print(kor_encode)   # 인코드 -> 벡터화됨
# print(eng_encode)   # 인코드 -> 벡터화됨
# print(kor_decode)   # 디코드 -> 다시 문자화됨 (스페셜 토큰은 살아있음)
# print(eng_decode)   # 디코드 -> 다시 문자화됨 (스페셜 토큰은 살아있음)


# 실제로 해보기

# 데이터 불러오기
data_in_path = '/Users/beomjunkim/Programming/Winter_NLP/nlp_dataset/'
train_data = pd.read_csv(data_in_path + 'ratings_train.txt',
                         header = 0, delimiter ='\t', quoting = 3)


input_ids = []
attention_masks = []
token_type_ids = []
train_data_labels = []

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    #sub: substitute; 괄호 안의 있는 것들을 대체하라. (무엇으로?)
    # [^...]: ^는 부정을 의미. 즉, ...의 것들을 제외한 모든것들을 대체하라. (무엇으로?)
    # 가-힣ㄱ-ㅎㅏ-ㅣ\\s: 가~힣, ㄱ~ㅎ, ㅏ~ㅣ, \s (모든 글자 및 자/모음, 공백(\s))
    # " "로 대체하라.
    # sent: sent를...
    # 결론: sent를 입력받아, 가~힣, ㄱ~ㅎ, ㅏ~ㅣ, 공백이 아닌 모든 것들을, " "로 대체한다.
    return sent_clean

for train_sent, train_label, in zip(train_data['document'], train_data['label']):
    try:
        input_id, attention_mask, token_type_id = \
        bert_tokenizer(clean_text(train_sent), MAX_LEN=15)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        train_data_labels.append(train_label)

    except Exception as e:
        print(e)
        print(train_sent)
        pass

train_movie_input_ids = np.array(input_ids, dtype = int)
train_movie_input_masks = np.array(attention_masks, dtype = int)
train_movie_type_ids = np.array(token_type_ids, dtype = int)
train_movie_inputs = (train_movie_input_ids, train_movie_input_masks, train_movie_type_ids)

train_data_labels = np.asarray(train_data_labels, dtype = np.int32) # 정답 토크나이징 리스트
print(f"# of sents: {len(train_movie_input_ids)}, # of labels: {len(train_data_labels)}")