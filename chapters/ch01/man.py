class Man:
    def __init__(self, name: str):
        self.name = name
        print("[Man] __init__")

    def hello(self):
        print("[Man] Hello, " + self.name + "!")

    def goodbye(self):
        print("[Man] Good-bye, " + self.name + ".")


man = Man("World")
man.hello()
man.goodbye()
