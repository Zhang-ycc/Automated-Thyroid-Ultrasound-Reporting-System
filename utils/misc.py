def boolean_to_zh_cn_is(b: bool) -> str:
    return '是' if b else '不是'

def boolean_to_zh_cn_exist(b: bool) -> str:
    return '存在' if b else '不存在'

def exist_to_calcification(x: int) -> str:
    calcifications = ['不存在结晶和钙化', '存在钙化', '存在结晶']
    return calcifications[x]

def en_to_zh_cn_composition(s: str) -> str:
    composition_dict = {
        'Cystic': '囊性',
        'Solid': '实性',
        'Mixed': '囊实性'
    }
    return composition_dict[s]

def en_to_zh_cn_echogenicity(s: str) -> str:
    echogenicity_dict = {
        'Anechoic': '无回声',
        'Hyperechoic': '高回声',
        'Isoechoic': '等回声',
        'Hypoechoic': '低回声',
        'Very Hypoechoic': '极低回声',
        'Echogenicity cannot be assessed': '无法判断回声'
    }
    return echogenicity_dict[s]


def boolean_to_zh_cn(b: bool) -> str:
    return '' if b else '不'