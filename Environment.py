from prettytable import PrettyTable, ALL
from copy import deepcopy as copy
from random import randint
import torch


class Game2048:
    def __init__(self, debug=False):
        self.table = torch.zeros(16, dtype=torch.int)
        self.debug = debug
        self.__is_played = True
        self.__zeros = 16
        self.__score = 0
        self.__step = 0
        self.__put_tile()

    def __check_played(self):
        if self.__zeros:
            return True
        for i in range(4):
            for j in range(3):
                if self.table[4 * i + j] == self.table[4 * i + j + 1]:
                    return True
        for i in range(3):
            for j in range(4):
                if self.table[4 * i + j] == self.table[4 * (i + 1) + j]:
                    return True
        return False

    def __put_tile(self, pos=None, num=None):
        if pos is None:
            pos = randint(1, self.__zeros)
        assert isinstance(pos, int)
        assert 1 <= pos <= self.__zeros

        if num is None:
            num = 2 if randint(1, 10) <= 9 else 4
        assert num == 2 or num == 4

        cnt = 0
        for i in range(4):
            for j in range(4):
                if self.table[4 * i + j] == 0:
                    cnt += 1
                if cnt == pos:
                    self.table[4 * i + j] = num
                    self.__zeros -= 1
                    if self.debug:
                        print(f'Поставили {num} на ({i + 1}, {j + 1})')
                    return i, j

    def __help_move(self, row, col, check, ori):
        assert check[0] == row or check[1] == col
        row_, col_ = check

        if (row_ != row or col != col_) and self.table[4 * row + col] != 0:
            if self.table[4 * row + col] == self.table[4 * row_ + col_]:
                self.table[4 * row_ + col_] *= 2
                self.table[4 * row + col] = 0
                self.__zeros += 1
                self.__score += self.table[4 * row_ + col_]
            else:
                if self.table[4 * row_ + col_] != 0:
                    check[0] += ori[0]
                    check[1] += ori[1]
                    row_, col_ = check
                if row_ != row or col != col_:
                    self.table[4 * row_ + col_] = self.table[4 * row + col]
                    self.table[4 * row + col] = 0

    def __left(self):
        for row in range(4):
            start = [row, 0]
            for col in range(4):
                self.__help_move(row, col, start, [0, 1])

    def __top(self):
        for col in range(4):
            start = [0, col]
            for row in range(4):
                self.__help_move(row, col, start, [1, 0])

    def __right(self):
        for row in range(4):
            start = [row, 3]
            for col in range(4)[::-1]:
                self.__help_move(row, col, start, [0, -1])

    def __bottom(self):
        for col in range(4):
            start = [3, col]
            for row in range(4)[::-1]:
                self.__help_move(row, col, start, [-1, 0])

    def __move(self, n):
        flag = True
        cnt = 0
        while flag:
            last_state = torch.clone(self.table)
            cnt += 1
            if n == 0:
                self.__left()
            elif n == 1:
                self.__top()
            elif n == 2:
                self.__right()
            elif n == 3:
                self.__bottom()
            if torch.all(last_state == self.table):
                flag = False
        return cnt > 1

    def get_reward(self):
        return self.__score
        # reward = 0
        # max_item = self.table.max()
        # for i in [0, 3]:
        #     for j in [0, 3]:
        #         if self.table[4*i + j] == max_item:
        #             reward += max_item
        # for i in range(4):
        #     for j in range(3):
        #         if self.table[4*i + j] > self.table[4*i + j + 1]:
        #             reward += self.table[4*i + j]
        #     for j in range(3)[::-1]:
        #         if self.table[4*i + j + 1] > self.table[4*i + j]:
        #             reward += self.table[4*i + j]
        # for j in range(4):
        #     for i in range(3):
        #         if self.table[4*i + j] > self.table[4*(i + 1) + j]:
        #             reward += self.table[4*i + j]
        #     for i in range(3)[::-1]:
        #         if self.table[4*(i + 1) + j] > self.table[4*i + j]:
        #             reward += self.table[4*i + j]
        # return reward

    def get_score(self):
        return self.__score

    def get_steps(self):
        return self.__step

    def is_played(self):
        return self.__is_played

    def get_copy_state(self):
        return copy(self)

    def action(self, n):
        """
          n = 0: left
          n = 1: top
          n = 2: right
          n = 3: bottom
        """
        assert 0 <= n < 4
        correct_move = self.__move(n)
        if correct_move:
            self.__step += 1
            self.__put_tile()
            self.__is_played = self.__check_played()
        return correct_move

    def next_states(self, n):
        correct_move = self.__move(n)
        if not correct_move:
            return None

        self.__step += 1
        for pos in range(self.__zeros):
            for num in [2, 4]:
                i, j = self.__put_tile(pos + 1, num)
                self.__is_played = self.__check_played()
                yield self
                self.table[4 * i + j] = 0
                self.__zeros += 1
        self.__put_tile()
        self.__is_played = self.__check_played()

    def print_state(self):
        table = PrettyTable(header=False, hrules=ALL)
        for i in range(4):
            row = []
            for j in range(4):
                row.append(self.table[4 * i + j].item())
            table.add_row(row)
        print(table)
        if self.debug:
            print(f'Количество нулей {self.__zeros}')
            print(f'Количество очков {self.__score}')
            print(f'Номер шага {self.__step}')


if __name__ == '__main__':
    env = Game2048(debug=True)
    while env.is_played():
        env.print_state()
        # act = int(input())
        act = randint(0, 3)
        env.action(act)
    env.print_state()
