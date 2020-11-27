import base64

if __name__ == '__main__':
    msg = "你是傻乎乎的"
    bytes_msg = msg.encode("utf-8")
    print(base64.b64encode(bytes_msg))
    msg = "5L2g5piv5YK75LmO5LmO55qE"
    print(base64.b64decode(msg).decode("utf-8"))