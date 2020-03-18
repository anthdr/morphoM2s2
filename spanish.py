



#data['regular'] = data['class'].apply(lambda x: '1' if 'regular' in x else '0')
#data['first'] = data['stem'].str.strip().str[0]
#data['first2'] = data['stem'].str.strip().str[1]
#data['first3'] = data['stem'].str.strip().str[2]
#data['first4'] = data['stem'].str.strip().str[3]
#data['last4'] = data['stem'].str.strip().str[-4]
#data['last3'] = data['stem'].str.strip().str[-3]
#data['last2'] = data['stem'].str.strip().str[-2]
#data['last'] = data['stem'].str.strip().str[-1]

#cv = CountVectorizer(data['stem'],ngram_range=(2,2),analyzer='char_wb')
#count_vector=cv.fit_transform(data['stem'])

# CountVectorizer character 2-grams with word boundaries
