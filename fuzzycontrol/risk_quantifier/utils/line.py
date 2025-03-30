class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def __call__(self, x):
        return self.m * x + self.b