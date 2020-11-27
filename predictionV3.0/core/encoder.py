def enctry(s, k):
    encry_str = ""
    for i, j in zip(s, k):
        temp = str(ord(i) + ord(j)) + '_'
        encry_str = encry_str + temp
    return encry_str


def dectry(p, k):
    dec_str = ""
    for i, j in zip(p.split("_")[:-1], k):
        temp = chr(int(i) - ord(j))
        dec_str = dec_str + temp
    return dec_str


if __name__ == '__main__':
    k = 'hanbaowang'
    data = "i love u"
    print("原始数据为：", data)
    enc_str = enctry(data, k)
    print("加密数据为：", enc_str)
    dec_str = dectry(enc_str, k)
    print("解密数据为：", dec_str)
