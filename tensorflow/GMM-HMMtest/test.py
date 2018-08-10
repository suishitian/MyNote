
a=5
def test():
    global a
    a = a +1
    print("a=",a)
print(a)
test()
print(a)