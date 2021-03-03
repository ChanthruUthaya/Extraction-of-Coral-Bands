
class Modify:
    def __init__(self):
        self.counter = 5

    def modify(self, x):
        x += self.counter
        return x




if __name__ == '__main__':
    num = 12
    mod = Modify()
    num = mod.modify(num)
    print(num)
