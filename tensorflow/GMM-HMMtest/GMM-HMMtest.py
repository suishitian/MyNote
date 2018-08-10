## -*- coding: utf-8 -*-
import numpy as np


class HMM:
    def __init__(self, Ann, Bnm, pi1n):
        self.A = np.array(Ann)     # 状态转移概率 NxN
        self.B = np.array(Bnm)     # 观测概率矩阵 NxM
        self.pi = np.array(pi1n)   # 初始状态概率向量，1xN
        self.N = self.A.shape[0]  #第一维长度，N种状态
        self.M = self.B.shape[1]  #第二维长度，M种观测结果

    def printhmm(self):
        print("==================================================")
        print("HMM content: N =", self.N, ",M =", self.M)
        for i in range(self.N):
            if i == 0:
                print("hmm.A ", self.A[i, :], " hmm.B ", self.B[i, :])
            else:
                print("      ", self.A[i, :], "       ", self.B[i, :])
        print("hmm.pi", self.pi)
        print("==================================================")

    # 函数名称：Forward *功能：前向算法估计参数 *参数:phmm:指向HMM的指针
    # T:观察值序列的长度 O:观察值序列
    # alpha:运算中用到的临时数组 pprob:返回值,所要求的概率
    def Forward(self, T, O, alpha, pprob):
        #     1. Initialization 初始化
        for i in range(self.N):
            # 初始状态为i且满足O(0)的概率，a|t=0（i）
            alpha[0, i] = self.pi[i] * self.B[i, O[0]]

        #     2. Induction 递归
        for t in range(T - 1):
            for j in range(self.N):
                sum = 0.0   # 中间量
                for i in range(self.N):
                    sum += alpha[t, i] * self.A[i, j]
                alpha[t + 1, j] = sum * self.B[j, O[t + 1]]
        #     3. Termination 终止
        sum = 0.0
        for i in range(self.N):
            sum += alpha[T - 1, i]    #   T-1时候的终止态
        pprob[0] *= sum   # sum是所有可能终止状态的加权和

    # 带修正的前向算法
    # scale[t] 走到第t步骤时，符合前向所有状态的概率，就是t时刻的N种状态的加权和
    def ForwardWithScale(self, T, O, alpha, scale, pprob):
        scale[0] = 0.0
        #     1. Initialization
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, O[0]]
            scale[0] += alpha[0, i]

        for i in range(self.N):
            alpha[0, i] /= scale[0]

        #     2. Induction
        for t in range(T - 1):
            scale[t + 1] = 0.0
            for j in range(self.N):
                sum = 0.0

                # alpha[t + 1, j] 使用了alpha[t, i]，是一个不断迭代的过程
                for i in range(self.N):
                    sum += alpha[t, i] * self.A[i, j]
                alpha[t + 1, j] = sum * self.B[j, O[t + 1]]
                # scale[t] t时刻 满足前向所有o（t）的概率
                scale[t + 1] += alpha[t + 1, j]
            for j in range(self.N):
                # alpha[t, j]是t时刻，此状态为j的概率可能性
                alpha[t + 1, j] /= scale[t + 1]

        #     3. Termination
        # 所有t时间的信息熵？？？？？？   为了求取极大似然函数做准备？
        for t in range(T):
            pprob[0] += np.log(scale[t])  #np.log以e为底

    # 函数名称：Backward * 功能:后向算法估计参数 * 参数:phmm:指向HMM的指针
    # T:观察值序列的长度 O:观察值序列
    # beta:运算中用到的临时数组 pprob:返回值，所要求的概率
    def Backword(self, T, O, beta, pprob):
        #     1. Intialization
        for i in range(self.N):
            # 最后一个为i时，P（o(t=T)| iT=q(i))=1 ？？？？
            beta[T - 1, i] = 1.0
        #     2. Induction
        for t in range(T - 2, -1, -1):   # 从最后一个往前数到第0个（start，stop，step）
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += self.A[i, j] * self.B[j, O[t + 1]] * beta[t + 1, j]
                beta[t, i] = sum

        #     3. Termination
        pprob[0] = 0.0
        # 所有满足序列的状态和，pprob[0]是所有的初始状态
        for i in range(self.N):
            pprob[0] += self.pi[i] * self.B[i, O[0]] * beta[0, i]

    # 带修正的后向算法
    def BackwardWithScale(self, T, O, beta, scale):
        #     1. Intialization
        for i in range(self.N):
            beta[T - 1, i] = 1.0

        #     2. Induction
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += self.A[i, j] * self.B[j, O[t + 1]] * beta[t + 1, j]
                # 最终目的是求得每种状态占t时刻的概率大小，最终态无所谓的？
                beta[t, i] = sum / scale[t + 1]

    # Viterbi算法
    # 输入：A，B，pi,O 输出P(O|lambda)最大时Poptimal的路径I
    def viterbi(self, O):
        T = len(O)
        # 初始化
        delta = np.zeros((T, self.N), np.float)  # delta是针对t时刻，t+1时刻的最优路径的概率值
        phi = np.zeros((T, self.N), np.float)    # 记录
        I = np.zeros(T)
        for i in range(self.N):
            # 0时刻 初态*变成Bi（O1）时刻的概率
            delta[0, i] = self.pi[i] * self.B[i, O[0]]
            phi[0, i] = 0
        # 递推
        for t in range(1, T):
            for i in range(self.N):
                delta[t, i] = self.B[i, O[t]] * np.array([delta[t - 1, j] * self.A[j, i] for j in range(self.N)]).max()
                phi[t, i] = np.array([delta[t - 1, j] * self.A[j, i] for j in range(self.N)]).argmax()
        # 终结
        prob = delta[T - 1, :].max()  # T时刻的最大值对应的状态 是 最优路径
        I[T - 1] = delta[T - 1, :].argmax()  # 取得最大值时 T时刻的变量
        # 状态序列求取，从最大值往前倒推
        for t in range(T - 2, -1, -1):
            I[t] = phi[t + 1, I[t + 1]]
        return I, prob  # prob是最终时刻的状态

    # 计算gamma : 满足观测序列，在t时刻处于状态i的概率  前向概率*后向概率/总概率 TxN矩阵
    # 返回值 gamma[T, N] 二维列表
    def ComputeGamma(self, T, alpha, beta, gamma):
        for t in range(T):
            denominator = 0.0 # 分母
            for j in range(self.N):
                gamma[t, j] = alpha[t, j] * beta[t, j]
                denominator += gamma[t, j]
            for i in range(self.N):
                gamma[t, i] = gamma[t, i] / denominator  # 除以总概率

    # 计算sai(i,j) 为给定训练序列O和模型lambda时：
    # t时刻 i状态，t+1时刻 j状态 在两个时刻N*N种可能性下的概率
    # 返回值 xi[T, N, N]
    def ComputeXi(self, T, O, alpha, beta, gamma, xi):
        for t in range(T - 1):
            sum = 0.0
            # t时刻时候，把xi序列填满
            for i in range(self.N):
                for j in range(self.N):
                    # 前向x后向x i状态转移到j状态的概率aij x 下一个显示成O(t+1)的概率bj(o(t+1))
                    # t+1时刻是未知的，所以直接用beta[t + 1, j]
                    xi[t, i, j] = alpha[t, i] * beta[t + 1, j] * self.A[i, j] * self.B[j, O[t + 1]]
                    sum += xi[t, i, j]

            # 归一化
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] /= sum

    # Baum-Welch算法
    # 输入 L个观察序列O，初始模型：HMM={A,B,pi,N,M}
    def BaumWelch(self, L, T, O, alpha, beta, gamma):

        #  初试参数
        print("Baum-Welch")
        DELTA = 0.01
        round = 0
        flag = 1
        probf = [0.0] # 只有 probf[0]的怪胎
        delta = 0.0
        deltaprev = 0.0
        probprev = 0.0
        ratio = 0.0
        deltaprev = 10e-70

        #  初始化
        xi = np.zeros((T, self.N, self.N))           # 9x2x2
        pi = np.zeros((T), np.float)                 # 1x9
        denominatorA = np.zeros((self.N), np.float)         # 1x2 分母
        denominatorB = np.zeros((self.N), np.float)         # 1x2
        numeratorA = np.zeros((self.N, self.N), np.float)      # 2x2  分子
        numeratorB = np.zeros((self.N, self.M), np.float)      # 2x2  NxM
        scale = np.zeros((T), np.float)                # 1x9

        while True:
            probf[0] = 0
            # E - step
            # L个长度为T的观测序列，由于齐次性，忽略时间t参数对 lamda(A, B, pi)的影响
            for l in range(L):
                self.ForwardWithScale(T, O[l], alpha, scale, probf)
                self.BackwardWithScale(T, O[l], beta, scale)
                self.ComputeGamma(T, alpha, beta, gamma)
                self.ComputeXi(T, O[l], alpha, beta, gamma, xi)
                for i in range(self.N):
                    pi[i] += gamma[0, i]   # pi初始值+0时刻处于i的概率，用L个初态做修正
                    for t in range(T - 1):
                        # A和B有区别吗 ？？？？
                        denominatorA[i] += gamma[t, i]   # 1x2 满足观测序列，所有状态为i的概率
                        denominatorB[i] += gamma[t, i]   # 1x2
                    denominatorB[i] += gamma[T - 1, i]   # B加上了最后一个的状态

                    for j in range(self.N):
                        for t in range(T - 1):
                            numeratorA[i, j] += xi[t, i, j]  # 满足观测序列，从i到j的概率
                    for k in range(self.M):
                        for t in range(T):    # 所有序列中，满足的个数
                            if O[l][t] == k:  # 观测序列t中 我们正在考察的那个
                                numeratorB[i, k] += gamma[t, i]  # 由于i状态 导致 观测为k 的概率！

            # M - step
            # 重估状态转移矩阵A 和 观察概率矩阵B
            for i in range(self.N):
                self.pi[i] = 0.001 / self.N + 0.999 * pi[i] / L  # pi[i]在E-step算了 L次
                for j in range(self.N):
                    # 这里不用管，因为numeratorA[i, j] 和 denominatorA[i] 都算了 L次
                    self.A[i, j] = 0.001 / self.N + 0.999 * numeratorA[i, j] / denominatorA[i]
                    numeratorA[i, j] = 0.0  # 为了下一轮的迭代

                for k in range(self.M):
                    self.B[i, k] = 0.001 / self.M + 0.999 * numeratorB[i, k] / denominatorB[i]
                    numeratorB[i, k] = 0.0

                # 为了下一轮的迭代的初始化
                pi[i] = denominatorA[i] = denominatorB[i] = 0.0

            # Flag决定此次迭代，直接下一次迭代
            # 没有更新参数？ A[i, j]，B[i, k]，pi[i]在上面已经更新完了
            # 没有把更新的参数保存下来
            if flag == 1:      # 没有更新参数？
                flag = 0
                probprev = probf[0]
                ratio = 1
                continue

            delta = probf[0] - probprev  # 更新的大小
            ratio = delta / deltaprev   # 和上一次相比更新的速率
            probprev = probf[0]    # 上一次参数存储一轮
            deltaprev = delta      # 上一次更新大小存储下来吗。。。。
            round += 1

            # ratio 更新率小于阈值 0.01，结束EM迭代
            if ratio <= DELTA:
                print("Num iteration： ", round)
                break


if __name__ == "__main__":
    print("Python my HMM")

    A = [[0.8125, 0.1875], [0.2, 0.8]]
    B = [[0.875, 0.125], [0.25, 0.75]]
    pi = [0.5, 0.5]
    hmm = HMM(A, B, pi)    # N=2，2个状态； M=2，2种观测结果

    O = [[1, 0, 0, 1, 1, 0, 0, 0, 0],
         [1, 1, 0, 1, 0, 0, 1, 1, 0],
         [0, 0, 1, 1, 0, 0, 1, 1, 1]]
    L = len(O)     # L=3
    T = len(O[0])  # T=9，时间长度序列为9？
    alpha = np.zeros((T, hmm.N), np.float)   # 9x2
    beta = np.zeros((T, hmm.N), np.float)
    gamma = np.zeros((T, hmm.N), np.float)
    hmm.BaumWelch(L, T, O, alpha, beta, gamma)

    hmm.printhmm()