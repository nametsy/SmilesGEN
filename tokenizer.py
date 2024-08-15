class Tokenizer:
    def __init__(self):
        self.start = "^"
        self.end = "$"
        self.pad = ' '
        self.int_to_char = None
        self.char_to_int = None
        self.tokenlist = None

    def build_vocab(self):
        chars = []
        # 原子符号
        chars = chars + ['H', 'B', "C", 'c', 'N', 'n', 'O', 'o', 'P', 'S', 's', 'F', 'I']
        # 将Si替换为Q,Cl替换为R,Br替换为V
        chars = chars + ['Q', 'R', 'V', 'Y', 'Z', 'G', 'T', 'U']
        # 将氢: H2替换为W,H3替换为X
        chars = chars + ['[', ']', '+', 'W', 'X']
        # 化学键,连接 - = # . / @ \
        chars = chars + ['-', '=', '#', '.', '/', '@', '\\']
        # 代表分支
        chars = chars + ['(', ')']
        # 环
        chars = chars + ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        # 编码列表 [pad,start,end,chars]
        self.tokenlist = [self.pad, self.start, self.end] + list(chars)
        # 原子字符 映射到 索引
        self.char_to_int = {c: i for i, c in enumerate(self.tokenlist)}
        # 索引 映射到 原子字符
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}

    @property
    def vocab_size(self):
        return len(self.int_to_char)

    # 编码,由字符串编码为索引列表,返回 数字形式
    def encode(self, smi):
        smi = smi.replace('Si', 'Q')
        smi = smi.replace('Cl', 'R')
        smi = smi.replace('Br', 'V')
        smi = smi.replace('Pt', 'Y')
        smi = smi.replace('Se', 'Z')
        smi = smi.replace('Li', 'T')
        smi = smi.replace('As', 'U')
        smi = smi.replace('Hg', 'G')
        smi = smi.replace('H2', 'W')
        smi = smi.replace('H3', 'X')
        return ([self.char_to_int[self.start]] +
                [self.char_to_int[s] for s in smi] +
                [self.char_to_int[self.end]])

    # 解码,由数字列表,编码为字符串,返回 Smiles字符串
    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords])
        smi = smi.replace('W', 'H2')
        smi = smi.replace('X', 'H3')
        smi = smi.replace('Q', 'Si')
        smi = smi.replace('R', 'Cl')
        smi = smi.replace('V', 'Br')
        smi = smi.replace('Y', 'Pt')
        smi = smi.replace('Z', 'Se')
        smi = smi.replace('T', 'Li')
        smi = smi.replace('U', 'As')
        smi = smi.replace('G', 'Hg')
        return smi

    @property
    def n_tokens(self):
        return len(self.int_to_char)


def vocabulary(args):
    tokenizer = Tokenizer()
    tokenizer.build_vocab()

    return tokenizer


if __name__ == '__main__':
    tokenize = Tokenizer()
    tokenize.build_vocab()
    print('\n')
    print('Vocabulary Information:')
    print('=' * 50)
    print(tokenize.char_to_int)
    print(tokenize.int_to_char)
    print('=' * 50)
