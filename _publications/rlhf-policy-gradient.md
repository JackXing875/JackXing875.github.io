---
layout: note
title: "RLHF 数学笔记：从策略梯度到 PPO 与 GRPO"
collection: publications
category: manuscripts
permalink: /publication/rlhf-policy-gradient/
date: 2026-10-17 21:18:00 +0800
display_date: "17 Oct 2026"
excerpt: "从轨迹概率、策略梯度与优势估计出发，逐步推导 GAE、TRPO、PPO 和 GRPO，再连接 RLOO、DAPO、Dr. GRPO、GSPO 与 DPO，理解后训练算法中的采样、裁剪与归一化。"
---

PPO、GRPO、RLOO 等方法的公式看起来差别很大，但它们都围绕几件事展开：如何从生成结果构造梯度，如何减少估计的噪声，如何复用已经采集的样本，以及如何避免一次更新改变策略过多。把这些问题分开，再把推导接起来，比记住一张算法缩写表更有用。

本文以有限长度的自回归文本生成为主要场景。先建立策略梯度的数学基础，再推导 PPO 与 GRPO；随后用 RLOO、Dr. GRPO、DAPO、GSPO 对照它们改动的具体位置，最后从同一个 KL 正则目标推到 DPO。示例数字只用于核对公式，不是模型实验结果。

## 1. RLHF 的优化问题从哪里来

### 1.1 从偏好比较到标量奖励

设输入为 $$x$$，完整回答为 $$y$$，偏好数据包含三元组 $$(x,y_w,y_l)$$：在同一个输入下，标注者更喜欢 $$y_w$$。一种常用建模方式是 Bradley–Terry 模型：

$$
P_\phi(y_w\succ y_l\mid x)
=\frac{\exp R_\phi(x,y_w)}{\exp R_\phi(x,y_w)+\exp R_\phi(x,y_l)}
=\sigma\!\left(R_\phi(x,y_w)-R_\phi(x,y_l)\right),
$$

其中 $$\sigma(z)=1/(1+e^{-z})$$。第二个等号来自分子分母同时除以 $$\exp R_\phi(x,y_w)$$。对每个已观察到的偏好标签取负对数，得到奖励模型的拟合目标：

$$
\mathcal L_{\mathrm{RM}}(\phi)
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\log\sigma\!\left(R_\phi(x,y_w)-R_\phi(x,y_l)\right).
$$

它只约束两个回答的分数差。把同一输入下所有回答的分数同时加上 $$c(x)$$，偏好概率不变。因此，奖励模型的绝对分数零点没有天然意义；模型输出 6 分，并不意味着存在统一的“六分回答”标准。

在随后的一轮策略优化里，通常把 $$\phi$$ 固定，只更新语言模型参数 $$\theta$$。奖励也可以来自测试程序、数学答案检查器，或者混合的反馈规则。**RLHF、RLVR 主要说明反馈来自哪里；PPO、GRPO 主要说明怎样利用反馈更新策略。** 不能仅凭优化器的名字判断奖励来源。

### 1.2 策略、参考模型与序列目标

自回归策略把一个回答的概率分解为

$$
\pi_\theta(y\mid x)
=\prod_{t=0}^{T-1}\pi_\theta(y_t\mid x,y_{<t}),
\qquad
\log\pi_\theta(y\mid x)
=\sum_{t=0}^{T-1}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

$$y_{<t}$$ 表示第 $$t$$ 步之前已经生成的所有 token；$$\theta$$ 是策略网络的参数。乘积来自概率的链式法则，不要求不同 token 独立。

这里 $$T$$ 是回答长度，包含约定的结束动作；有限长度建模也可以通过终止后的吸收状态统一处理。输入 token 不属于策略在本轮生成的动作，通常不会放入回答的策略梯度求和。

令输入分布为 $$\mu(x)$$，参考策略为固定的 $$\pi_{\mathrm{ref}}$$，则经典的 KL 正则化目标是

$$
J_\beta(\theta)
=\mathbb E_{x\sim\mu}\left[
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}R(x,y)
-\beta D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\right)
\right].
$$

这里 $$\mathbb E_{z\sim p}[f(z)]=\sum_z p(z)f(z)$$ 表示按分布 $$p$$ 对数值 $$f(z)$$ 做加权平均；连续变量时求和替换为积分。双层期望意味着先采输入，再按该输入的条件策略采回答。KL 散度定义为

$$
D_{\mathrm{KL}}(p\Vert q)=\sum_zp(z)\log\frac{p(z)}{q(z)}.
$$

它比较两个完整分布，且具有方向性；本文使用自然对数。若 $$p(z)>0$$ 而 $$q(z)=0$$，对应项为无穷大。按这个定义展开：

$$
J_\beta(\theta)
=\mathbb E_{x\sim\mu,y\sim\pi_\theta}
\left[R(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right].
$$

$$\beta>0$$ 控制奖励与分布偏离之间的权衡。参考策略通常取训练前的 SFT 模型。它与后文采样用的旧策略 $$\pi_{\mathrm{old}}$$ 不同：参考策略定义希望保留的行为锚点，旧策略记录这一批数据是由谁生成的。

上述 KL 方向也很重要：期望取在当前策略下。交换两个分布并不是符号上的等价变形。

为了先看清策略梯度本身，接下来暂时把奖励看成不显式依赖 $$\theta$$ 的固定函数；推导到 RLHF 的实际更新时，再单独处理 KL 项。



## 2. 从一个明确的期望目标出发

设一条有限轨迹为

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_{T-1},a_{T-1},r_{T-1},s_T).
$$

这里 $$s_t$$ 是采取第 $$t$$ 个动作之前的状态，$$a_t$$ 是动作，$$r_t$$ 是该动作之后得到的奖励，$$s_T$$ 是终止状态。若任务有固定时限而相同观测在不同时间有不同含义，应把时间 $$t$$ 纳入状态。语言模型的状态可以直接取为“提示词和已生成的全部前缀”，因此自然包含生成位置：

$$
s_t=(x,y_{<t}),\qquad a_t=y_t,\qquad s_{t+1}=(x,y_{\le t}).
$$

本文先采用 **有限回合、$$\gamma=1$$** 的约定：

$$
R(\tau)=\sum_{t=0}^{T-1}r_t,
\qquad
J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)].
\tag{1}
$$

如果只有完整回答得到评分 $$R(x,y)$$，可以令中间奖励为零、最后一步奖励为 $$R(x,y)$$。变长回答在 EOS 处结束；为了统一成固定最大长度的符号，也可以在 EOS 后进入只有一个虚拟动作、奖励恒为零的吸收状态。

这里先假设初始状态分布和环境机制不依赖 $$\theta$$，奖励函数也没有对 $$\theta$$ 的显式依赖。参考模型 KL 正则涉及的显式参数依赖，需要在相应正则化目标中另行处理。

记初始状态分布为 $$\mu$$，状态转移与奖励的联合分布为 $$P(s_{t+1},r_t\mid s_t,a_t)$$。轨迹概率分解为

$$
p_\theta(\tau)
=
\mu(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)
P(s_{t+1},r_t\mid s_t,a_t).
\tag{2}
$$

对语言模型，给定前缀和下一个 token，新的前缀已经确定，因此状态转移通常是确定性的。我们仍保留一般环境写法，以看清哪些项会被求导。

## 3. 策略梯度为什么出现 log probability

假设可以交换求导与求和或积分，并且策略具有适当的共同支持集。从式 (1) 出发：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\nabla_\theta\sum_\tau p_\theta(\tau)R(\tau)\\
&=\sum_\tau \nabla_\theta p_\theta(\tau)R(\tau)\\
&=\sum_\tau p_\theta(\tau)
   \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}R(\tau)\\
&=\mathbb E_{\tau\sim p_\theta}
  \left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right].
\end{aligned}
\tag{3}
$$

第三行只是在每项乘上再除以 $$p_\theta(\tau)$$；第四行使用
$$\nabla\log p=(\nabla p)/p$$。这个恒等式通常称为 likelihood-ratio trick 或 score-function identity。它让我们不必对离散采样出来的 token 本身求导。

再把式 (2) 取对数：

$$
\log p_\theta(\tau)
=
\log\mu(s_0)
+\sum_t\log\pi_\theta(a_t\mid s_t)
+\sum_t\log P(s_{t+1},r_t\mid s_t,a_t).
$$

初始分布和环境机制都与策略参数无关，所以它们对 $$\theta$$ 的导数为零，得到

$$
\nabla_\theta\log p_\theta(\tau)
=
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\tag{4}
$$

因此

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].}
\tag{5}
$$

轨迹内的状态当然会受此前动作影响，但式 (2) 已经通过轨迹概率完整地计入这种影响。式 (4) 不是在假定状态分布不变，也不要求沿离散状态转移反向传播。

### 3.1 为什么过去的奖励可以删掉

令

$$
z_t=\nabla_\theta\log\pi_\theta(a_t\mid s_t),
\qquad
G_t=\sum_{k=t}^{T-1}r_k.
$$

在动作 $$a_t$$ 之前的完整历史 $$H_t$$ 给定时，

$$
\begin{aligned}
\mathbb E[z_t\mid H_t]
&=\sum_a\pi_\theta(a\mid s_t)
      \nabla_\theta\log\pi_\theta(a\mid s_t)\\
&=\sum_a\nabla_\theta\pi_\theta(a\mid s_t)\\
&=\nabla_\theta\sum_a\pi_\theta(a\mid s_t)\\
&=\nabla_\theta 1=0.
\end{aligned}
\tag{6}
$$

对任何 $$k<t$$，奖励 $$r_k$$ 已经包含在 $$H_t$$ 中，因此

$$
\mathbb E[r_k z_t]
=
\mathbb E\left[r_k\mathbb E[z_t\mid H_t]\right]
=0.
\tag{7}
$$

把总奖励拆成过去和未来，过去部分的期望贡献为零，于是式 (5) 化为

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E\left[
\sum_{t=0}^{T-1}G_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].}
\tag{8}
$$

这就是 reward-to-go 形式。它使用的因果事实非常具体：现在的动作不能改变已经发生的奖励。删除这些期望为零、却会给有限样本带来噪声的项，通常有助于降低估计方差。

### 3.2 折扣系数最容易被省略的一项

如果改为从初始时刻定义折扣目标

$$
J_\gamma(\theta)
=
\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],
\qquad
G_t^\gamma=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,
$$

同样的推导得到的是

$$
\boxed{
\nabla_\theta J_\gamma(\theta)
=
\mathbb E\left[
\sum_{t=0}^{T-1}\gamma^tG_t^\gamma
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].}
\tag{9}
$$

原因是 $$\sum_{k=t}^{T-1}\gamma^k r_k=\gamma^tG_t^\gamma$$。外面的 $$\gamma^t$$ 不能仅凭“回报内部已有折扣”就删掉。文献里没有这个外因子的写法，可能使用了折扣状态访问分布，也可能把折扣作为近似梯度中的方差控制手段；它们需要各自说明约定。本文的主线取 $$\gamma=1$$，避免把这些不同约定混为一谈。GAE 原论文明确区分了未折扣目标与用于估计的折扣近似。[GAE 原论文，§2](https://arxiv.org/pdf/1506.02438)

## 4. 基线、价值函数与优势

### 4.1 基线为什么不改变期望梯度

给任意只依赖动作前状态的函数 $$b(s_t)$$，式 (6) 直接给出

$$
\mathbb E[b(s_t)z_t]
=
\mathbb E_{s_t}\left[b(s_t)\mathbb E[z_t\mid s_t]\right]=0.
$$

所以

$$
\nabla_\theta J(\theta)
=
\mathbb E\left[
\sum_t\bigl(G_t-b(s_t)\bigr)z_t
\right].
\tag{10}
$$

减去基线改变的是每个样本的数值与噪声，不改变期望方向。关键限制是：基线在给定动作前信息后，不能再偷偷依赖本次采样动作。训练时还应把 actor 更新中的基线和优势当作固定权重，不沿它们反向传播。

一个常见细节是，把包含当前样本自己的批均值直接当基线，并不自动满足独立性。以同一提示词下的独立回答为例，令 $$\bar R=B^{-1}\sum_jR_j$$，则

$$
\mathbb E\left[\frac1B\sum_i(R_i-\bar R)z_i\right]
=
\left(1-\frac1B\right)\mathbb E[Rz].
$$

这里 $$z_i=\nabla_\theta\log\pi_\theta(y_i\mid x)$$ 是第 $$i$$ 个完整回答的 score。来自 $$j=i$$ 的项不能按独立基线消掉；来自其他样本的项才可以。这也解释了 leave-one-out 基线的一项数学动机。

### 4.2 $$V$$、$$Q$$、$$A$$ 分别条件化了什么

对固定策略 $$\pi$$，定义

$$
V^\pi(s_t)=\mathbb E_\pi[G_t\mid s_t],
\qquad
Q^\pi(s_t,a_t)=\mathbb E_\pi[G_t\mid s_t,a_t],
\tag{11}
$$

以及

$$
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t).
\tag{12}
$$

$$V$$ 的条件是“已经到达这个状态”；$$Q$$ 额外规定“现在先采取这个动作”；两者之后都继续遵循同一策略 $$\pi$$。因此

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a),
\qquad
\sum_a\pi(a\mid s)A^\pi(s,a)=0.
\tag{13}
$$

优势衡量的是一个动作相对于当前策略在同一状态下平均水平的增益。它不是动作的绝对分数。

利用条件期望，把式 (8) 中的 $$G_t$$ 替换成 $$Q^\pi(s_t,a_t)$$：

$$
\mathbb E[G_tz_t]
=
\mathbb E\left[\mathbb E[G_t\mid s_t,a_t]z_t\right]
=
\mathbb E[Q^\pi(s_t,a_t)z_t].
$$

再减去 $$V^\pi(s_t)$$ 这个动作无关基线，就得到

$$
\boxed{
\nabla_\theta J(\theta)
=
\mathbb E_{\pi_\theta}\left[
\sum_t A^{\pi_\theta}(s_t,a_t)
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].}
\tag{14}
$$

$$V^\pi$$ 是一个自然的基线，但不应把它写成“任何情形下方差最小的基线”。若仅在给定状态下最小化单步随机向量 $$(G-b)z$$ 的二阶矩，最优标量基线为

$$
b^*(s)=
\frac{\mathbb E[G\|z\|^2\mid s]}{\mathbb E[\|z\|^2\mid s]},
$$

前提是分母非零。$$V^\pi(s)=\mathbb E[G\mid s]$$ 没有其中的梯度范数权重；它的优势在于含义清楚、可复用，并能配合 Bellman 关系做自举估计。

### 4.3 一个两动作例子

同一状态下有动作 $$a$$、$$b$$，策略概率分别为 $$0.3$$、$$0.7$$，之后的期望回报分别为 $$5$$、$$1$$。于是

$$
V(s)=0.3\times5+0.7\times1=2.2,
$$

$$
A(s,a)=2.8,\qquad A(s,b)=-1.2.
$$

检查零均值：$$0.3\times2.8+0.7\times(-1.2)=0$$。若用单个 logit $$u$$ 表示 $$\pi(a)=\sigma(u)$$，则

$$
\partial_u\log\pi(a)=1-0.3=0.7,
\qquad
\partial_u\log\pi(b)=-0.3.
$$

优势形式的期望梯度是

$$
0.3\times2.8\times0.7
+
0.7\times(-1.2)\times(-0.3)
=0.84.
$$

直接对 $$J(u)=5\sigma(u)+1(1-\sigma(u))$$ 求导也得到
$$4\times0.3\times0.7=0.84$$。正优势动作的正贡献和负优势动作的负 log-score 相乘，都会推动较好的动作概率上升。

## 5. 从 Bellman 关系到 TD 与 GAE

### 5.1 Bellman 关系把未来拆成一步和剩余部分

以下为了呈现通用估计器，在 $$V,Q$$ 中允许使用 $$\gamma\in[0,1]$$；主线 RLHF 情形仍可直接代入 $$\gamma=1$$。定义 $$G_t^\gamma=r_t+\gamma G_{t+1}^\gamma$$，由条件期望的塔式法则得到

$$
Q^{\pi,\gamma}(s_t,a_t)
=
\mathbb E\left[
 r_t+\gamma V^{\pi,\gamma}(s_{t+1})
 \mid s_t,a_t
\right].
\tag{15}
$$

因此，如果价值函数恰好准确，

$$
\begin{aligned}
\mathbb E\left[
 r_t+\gamma V^{\pi,\gamma}(s_{t+1})-V^{\pi,\gamma}(s_t)
 \mid s_t,a_t\right]
&=Q^{\pi,\gamma}(s_t,a_t)-V^{\pi,\gamma}(s_t)\\
&=A^{\pi,\gamma}(s_t,a_t).
\end{aligned}
\tag{16}
$$

实践中用学习得到的 $$V_\psi$$ 代替未知的真实价值，定义 TD 残差

$$
\delta_t=r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t).
\tag{17}
$$

它比较“当前预测”与“观察到的一步奖励加下一状态预测”。只有当后续价值预测满足相应准确性条件时，它才是无偏优势估计；把 $$V_\psi$$ 写成神经网络并不会自动保证这一点。

### 5.2 多步估计来自望远镜求和

令 $$n\le T-t$$。把连续 $$n$$ 个 TD 残差按折扣加起来：

$$
\begin{aligned}
\sum_{\ell=0}^{n-1}\gamma^\ell\delta_{t+\ell}
&=\sum_{\ell=0}^{n-1}\gamma^\ell r_{t+\ell}
 +\sum_{\ell=0}^{n-1}\gamma^{\ell+1}V_\psi(s_{t+\ell+1})
 -\sum_{\ell=0}^{n-1}\gamma^\ell V_\psi(s_{t+\ell})\\
&=\sum_{\ell=0}^{n-1}\gamma^\ell r_{t+\ell}
 +\gamma^n V_\psi(s_{t+n})-V_\psi(s_t).
\end{aligned}
\tag{18}
$$

第二行之所以只剩两个价值项，是因为中间每个 $$V_\psi(s_{t+k})$$ 都出现了一次正号和一次负号，系数同为 $$\gamma^k$$，逐一抵消。由此定义 $$n$$ 步优势估计：

$$
\widehat A_t^{(n)}
=
\sum_{\ell=0}^{n-1}\gamma^\ell r_{t+\ell}
+\gamma^nV_\psi(s_{t+n})-V_\psi(s_t).
\tag{19}
$$

$$n=1$$ 就是 TD 残差；$$n=T-t$$ 且 $$s_T$$ 是真正终止状态、$$V_\psi(s_T)=0$$ 时，就得到完整实测回报减去当前价值基线。增加 $$n$$ 用更多真实奖励替代远端价值预测，通常减少自举误差的影响，但会引入更多轨迹随机性。方差是否单调变化仍取决于具体问题。

### 5.3 有限回合的 GAE 权重必须补齐尾项

记剩余步数 $$H=T-t$$。定义有限回合的加权组合

$$
\widehat A_t^{\mathrm{GAE}}
=
(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}\widehat A_t^{(n)}
+
\lambda^{H-1}\widehat A_t^{(H)},
\qquad \lambda\in[0,1].
\tag{20}
$$

这些权重之和等于

$$
(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}+\lambda^{H-1}=1.
$$

最后一项收集了原本会分配给更长回报的所有剩余权重；如果简单把无限几何级数截断却不补尾项，有限回合的定义就变了。

现在用式 (18) 展开每个 $$\widehat A_t^{(n)}$$。对于固定的 $$\delta_{t+\ell}$$，它出现在所有 $$n\ge\ell+1$$ 的项里，其总系数为

$$
\begin{aligned}
\gamma^\ell\left[
 (1-\lambda)\sum_{n=\ell+1}^{H-1}\lambda^{n-1}
 +\lambda^{H-1}\right]
&=\gamma^\ell\left[\lambda^\ell-\lambda^{H-1}+\lambda^{H-1}\right]\\
&=(\gamma\lambda)^\ell.
\end{aligned}
$$

所以最终得到

$$
\boxed{
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{\ell=0}^{T-t-1}(\gamma\lambda)^\ell\delta_{t+\ell}.}
\tag{21}
$$

它可以反向递推计算：

$$
\widehat A_T=0,
\qquad
\widehat A_t=\delta_t+\gamma\lambda\widehat A_{t+1}.
\tag{22}
$$

GAE 的原始构造是多步估计的指数加权组合；上面的有限回合推导补出了在实现中需要明确处理的尾项。[GAE 原论文，§3](https://arxiv.org/pdf/1506.02438)

### 5.4 $$\lambda$$ 两个端点与偏差的来源

- $$\lambda=0$$：$$\widehat A_t=\delta_t$$，只用一步观察，其余依赖价值自举。
- $$\lambda=1$$：$$\widehat A_t=G_t^\gamma-V_\psi(s_t)$$，前提是完整回合和正确的零终止值。

第二种情形中，即使 $$V_\psi$$ 不准确，$$-V_\psi(s_t)$$ 仍然只是动作无关基线，因此不会给相应的策略梯度增加基线偏差。但这不等于说该数值本身就是无偏的真实优势：其条件均值是 $$Q^{\pi,\gamma}-V_\psi$$，而非必然等于 $$Q^{\pi,\gamma}-V^{\pi,\gamma}$$。

更精确地，令价值误差为

$$
e_t=V_\psi(s_t)-V^{\pi,\gamma}(s_t).
$$

把近似价值产生的 GAE 减去真实价值产生的 GAE：

$$
\begin{aligned}
\widehat A_t(V_\psi)-\widehat A_t(V^{\pi,\gamma})
&=\sum_{\ell=0}^{H-1}(\gamma\lambda)^\ell(\gamma e_{t+\ell+1}-e_{t+\ell})\\
&=-e_t
+\gamma(1-\lambda)\sum_{k=1}^{H-1}(\gamma\lambda)^{k-1}e_{t+k}
+\gamma^H\lambda^{H-1}e_T.
\end{aligned}
\tag{23}
$$

真正终止时 $$e_T=0$$。第一项只是当前状态基线误差，在策略梯度的期望里消失；中间那些未来状态误差可能依赖当前动作，通常不能消掉。当 $$\lambda=1$$ 时，中间项也消失。这展示了 $$\lambda$$ 为什么会控制自举偏差，而不只是一个凭经验添加的平滑参数。

如果 $$V_\psi=V^{\pi,\gamma}$$，任意 $$\lambda$$ 的这些误差项都为零。所谓“小 $$\lambda$$ 更有偏”指的是一般的近似价值情形，并不是一个无条件定理。

### 5.5 数值例子：终点奖励怎样传回较早 token

考虑三步回答，$$\gamma=1$$，奖励和价值预测为

$$
(r_0,r_1,r_2)=(0,0,1),
\qquad
(V_0,V_1,V_2,V_3)=(0.2,0.4,0.7,0).
$$

TD 残差为

$$
\delta_0=0+0.4-0.2=0.2,
\quad
\delta_1=0+0.7-0.4=0.3,
\quad
\delta_2=1+0-0.7=0.3.
$$

若 $$\lambda=0.9$$，反向递推得到

$$
\widehat A_2=0.3,
\qquad
\widehat A_1=0.3+0.9\times0.3=0.57,
$$

$$
\widehat A_0=0.2+0.9\times0.57=0.713.
$$

也可以从多步组合检查：对 $$t=0$$，

$$
\widehat A_0^{(1)}=0.2,
\quad
\widehat A_0^{(2)}=0.5,
\quad
\widehat A_0^{(3)}=0.8,
$$

有限 GAE 权重分别是 $$0.1,0.09,0.81$$，于是

$$
0.1\times0.2+0.09\times0.5+0.81\times0.8=0.713.
$$

如果 $$\lambda=1$$，三个优势变成 $$(0.8,0.6,0.3)$$，恰好等于完整回报 $$1$$ 分别减去 $$(0.2,0.4,0.7)$$。

价值网络可用固定的回报目标训练。例如使用

$$
\widehat R_t=\operatorname{stopgrad}(\widehat A_t+V_{\psi_{\mathrm{old}}}(s_t)),
\qquad
\frac12\mathbb E\left[(V_\psi(s_t)-\widehat R_t)^2\right].
\tag{24}
$$

此处 $$\widehat R_t$$ 是 GAE 对应的 $$\lambda$$-return 目标，$$\lambda<1$$ 时通常不等于原始 Monte Carlo 回报。用于生成目标的旧价值预测和用于拟合目标的当前预测也应区分。

### 5.6 真正终止与采样截断不能混淆

若 EOS 或任务定义的终止已发生，未来奖励确实不存在，应令终止价值为零。若只是 rollout 收集窗口结束，而数学任务本来还会继续，则通常需要用最后状态价值自举。把这种截断误当成真正终止，相当于把尚未观察到的未来回报错误设为零。是否应在最大长度处置零，取决于任务是否把该上限定义为真正结束，而非一个统一的软件习惯。

## 6. 为什么 PPO 需要旧策略和概率比

收集一批轨迹时，设实际采样策略是 $$\pi_{\mathrm{old}}$$。随后更新参数得到 $$\pi_\theta$$，但手中的样本仍来自 $$\pi_{\mathrm{old}}$$。在固定状态 $$s$$ 上，定义

$$
\rho_\theta(s,a)
=\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}.
\tag{25}
$$

只要新策略的相关支持被旧策略覆盖，重要性采样恒等式为

$$
\begin{aligned}
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[f(s,a)]
&=\sum_a\pi_\theta(a\mid s)f(s,a)\\
&=\sum_a\pi_{\mathrm{old}}(a\mid s)
  \frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}f(s,a)\\
&=\mathbb E_{a\sim\pi_{\mathrm{old}}}[\rho_\theta(s,a)f(s,a)].
\end{aligned}
\tag{26}
$$

**这条等式只修正给定状态下的动作分布。** 它没有把旧策略访问到的前缀分布变成新策略的前缀分布。若要对整条轨迹精确换分布，权重为

$$
\frac{p_\theta(\tau)}{p_{\mathrm{old}}(\tau)}
=
\prod_{t=0}^{T-1}\rho_t(\theta).
\tag{27}
$$

长序列中的概率比连乘很容易产生高方差。PPO 使用逐 token 概率比配合一个局部 surrogate，而不是声称逐 token 比率已经精确校正了所有轨迹分布变化。

实现中用 log probability 更稳定：

$$
\rho_t(\theta)=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\mathrm{old}}(a_t\mid s_t)
\right).
\tag{28}
$$

同一批样本的旧 log probability 应保持固定。若每个优化步都把分母替换成当前策略，数值比率会被反复重置为 $$1$$，也就失去了表示“相对生成时策略变化多少”的意义。

## 7. 从性能差异恒等式到 TRPO 的局部近似

本节的 $$T$$ 明确表示所有轨迹共同的确定性最大时域，不是某条样本随机生成出的回答长度。若实际回答提前结束，就按前文约定在其后补齐零奖励吸收状态，直到共同的 $$T$$；以下性能恒等式与误差界均使用这个约定。

### 7.1 先写出一个精确恒等式

回到有限回合、$$\gamma=1$$。让 $$d_t^\pi(s)=\Pr_\pi(s_t=s)$$。对任意新策略 $$\pi$$，有

$$
\boxed{
J(\pi)-J(\pi_{\mathrm{old}})
=
\sum_{t=0}^{T-1}
\mathbb E_{s_t\sim d_t^\pi,\,a_t\sim\pi}
\left[A^{\mathrm{old}}(s_t,a_t)\right].}
\tag{29}
$$

证明只需 Bellman 关系和望远镜求和：

$$
\begin{aligned}
\mathbb E_\pi\sum_t A^{\mathrm{old}}(s_t,a_t)
&=\mathbb E_\pi\sum_t
 [r_t+V^{\mathrm{old}}(s_{t+1})-V^{\mathrm{old}}(s_t)]\\
&=\mathbb E_\pi\sum_t r_t
 -\mathbb E_\mu V^{\mathrm{old}}(s_0)
 +\mathbb E_\pi V^{\mathrm{old}}(s_T)\\
&=J(\pi)-J(\pi_{\mathrm{old}}).
\end{aligned}
$$

第一行等式在对环境转移取期望后成立；最后一行使用同一初始分布及零终止价值。困难在于式 (29) 的状态分布 $$d_t^\pi$$ 也是未知新策略的分布。

### 7.2 Surrogate 具体近似了什么

把式 (29) 的新状态分布替换成旧状态分布，定义

$$
\begin{aligned}
L_{\mathrm{old}}(\theta)
&=J(\pi_{\mathrm{old}})
 +\sum_t\mathbb E_{s_t\sim d_t^{\mathrm{old}},\,a_t\sim\pi_\theta}
    [A^{\mathrm{old}}(s_t,a_t)]\\
&=J(\pi_{\mathrm{old}})
 +\mathbb E_{\tau\sim\pi_{\mathrm{old}}}
    \sum_t\rho_t(\theta)A^{\mathrm{old}}(s_t,a_t).
\end{aligned}
\tag{30}
$$

第二行才是对动作使用式 (26)。在 $$\theta=\theta_{\mathrm{old}}$$ 处，$$\rho_t=1$$，而且

$$
\nabla_\theta\rho_t(\theta)
=\rho_t(\theta)\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\tag{31}
$$

因此

$$
\left.\nabla_\theta L_{\mathrm{old}}(\theta)\right|_{\theta=\theta_{\mathrm{old}}}
=
\mathbb E_{\mathrm{old}}\sum_t
A^{\mathrm{old}}(s_t,a_t)
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big|_{\theta_{\mathrm{old}}}
=
\nabla J(\theta_{\mathrm{old}}).
\tag{32}
$$

也就是说，局部 surrogate 在旧策略处与真实目标的一阶变化一致。离旧策略越远，忽略状态访问分布变化的近似就越需要警惕。TRPO 论文的性能差异关系和局部 surrogate 正是围绕这一问题展开。[TRPO 原论文，§2–3](https://arxiv.org/pdf/1502.05477)

### 7.3 一个有限时域误差界，说明为何需要“小更新”

令

$$
\alpha=\max_s D_{\mathrm{TV}}\bigl(\pi_{\mathrm{old}}(\cdot\mid s),\pi(\cdot\mid s)\bigr),
\qquad
A_{\max}=\max_{s,a}|A^{\mathrm{old}}(s,a)|.
$$

这里总变差距离定义为 $$D_{\mathrm{TV}}(p,q)=\tfrac12\sum_a\lvert p(a)-q(a)\rvert$$，衡量两个离散分布在概率质量上的差别；$$\lVert v\rVert_1=\sum_s\lvert v(s)\rvert$$ 是绝对值之和。状态已包含时间。根据两策略逐步耦合的简单上界，第 $$t$$ 步的状态分布满足

$$
\|d_t^\pi-d_t^{\mathrm{old}}\|_1\le 2t\alpha.
$$

又因为 $$\sum_a\pi_{\mathrm{old}}(a\mid s)A^{\mathrm{old}}(s,a)=0$$，

$$
\left|\sum_a\pi(a\mid s)A^{\mathrm{old}}(s,a)\right|
=
\left|\sum_a(\pi-\pi_{\mathrm{old}})(a\mid s)A^{\mathrm{old}}(s,a)\right|
\le2\alpha A_{\max}.
$$

把这两个界代入真实性能差异与 surrogate 的差，得到

$$
\begin{aligned}
|J(\pi)-L_{\mathrm{old}}(\pi)|
&\le\sum_{t=0}^{T-1}(2t\alpha)(2\alpha A_{\max})\\
&=2A_{\max}T(T-1)\alpha^2.
\end{aligned}
\tag{33}
$$

这不是一个必须拿来设置实际步长的紧界，但它清楚说明：近似误差随策略变化的平方增长，还受到序列长度的放大。这里采用有限时域推导，不能把无限折扣理论中含 $$(1-\gamma)^{-2}$$ 的界直接代入 $$\gamma=1$$。

### 7.4 实际 TRPO 的平均 KL 约束

在样本上，以 $$\widehat A_t$$ 近似真实优势，TRPO 的典型形式是

$$
\max_\theta\ \widehat{\mathbb E}_t[\rho_t(\theta)\widehat A_t],
\quad\text{满足}\quad
\widehat{\mathbb E}_t
\left[D_{\mathrm{KL}}\bigl(
\pi_{\mathrm{old}}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t)
\bigr)\right]\le\delta.
\tag{34}
$$

这里是旧策略到新策略的 KL。理论中的最坏状态约束、真实优势和精确期望，到了实际算法中通常被平均 KL、估计优势和有限样本替代。因此不能把实践中的每次 TRPO 更新都表述成严格保证真实性能单调提升。

进一步在旧参数附近写 $$\Delta=\theta-\theta_{\mathrm{old}}$$。对 surrogate 做一阶展开、对 KL 做二阶展开，得到局部问题

$$
\max_\Delta g^\top\Delta,
\qquad
\frac12\Delta^\top F\Delta\le\delta,
\tag{35}
$$

其中 $$g$$ 是 surrogate 在旧参数处的梯度，$$F$$ 是 KL 在相等策略处的 Hessian，在标准正则条件下等于 Fisher 信息矩阵。Hessian 指函数的二阶偏导矩阵；Fisher 则度量参数变化引起的概率分布局部变化。拉格朗日函数的驻点满足

$$
g-\nu F\Delta=0
\quad\Longrightarrow\quad
\Delta=\nu^{-1}F^{-1}g.
$$

在 $$F$$ 正定且 $$g\ne0$$ 的条件下，将它代入约束边界，得到

$$
\Delta=
\sqrt{\frac{2\delta}{g^\top F^{-1}g}}F^{-1}g.
\tag{36}
$$

这解释了自然梯度方向从何而来。若 $$g=0$$，一阶局部模型没有可提升的方向，式 (36) 的归一化形式不适用。神经网络的 Fisher 可能奇异或病态，实际系统通常借助阻尼（例如使用 $$F+\eta I$$，$$\eta>0$$）、共轭梯度和线搜索近似求解，也可在适当条件下使用伪逆，而不显式存储完整的 $$F$$。

## 8. PPO 的裁剪究竟裁掉了什么

PPO-Clip 使用

$$
\boxed{
L^{\mathrm{clip}}(\theta)
=
\widehat{\mathbb E}_t\left[
\min\left(
\rho_t(\theta)\widehat A_t,
\operatorname{clip}(\rho_t(\theta),1-\varepsilon,1+\varepsilon)\widehat A_t
\right)
\right].}
\tag{37}
$$

它把每个样本的原始 surrogate 与裁剪版本取较小值。PPO 原论文将这种构造用于交替执行采样和多轮小批量优化。[PPO 原论文，§3、§5](https://arxiv.org/pdf/1707.06347)

这里 $$\widehat{\mathbb E}_t$$ 表示采样数据上的经验平均：若一批有 $$B$$ 个回答、第 $$i$$ 个回答有 $$T_i$$ 个有效 token，则全 token 平均为 $$\sum_{i,t}(\cdot)/N$$，其中 $$N=\sum_iT_i$$。固定批次上的这个共同缩放不改变梯度方向；若改为每个回答先除以自己的长度再平均，则会改变不同长度回答之间的相对权重，需要单独说明。

在一个优化批次内部，$$\pi_{\mathrm{old}}$$ 的 log probability 和 $$\widehat A_t$$ 都当作固定量。下面令 $$A=\widehat A_t$$、$$\rho=\rho_t$$，逐样本分析

$$
f_A(\rho)=\min\bigl(\rho A,\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A\bigr).
$$

### 8.1 正优势：只给向上奖励设置平台

当 $$A>0$$ 时，乘上 $$A$$ 保持不等号方向：

$$
f_A(\rho)=A\min(\rho,1+\varepsilon)
=
\begin{cases}
\rho A,&\rho\le1+\varepsilon,\\
(1+\varepsilon)A,&\rho>1+\varepsilon.
\end{cases}
\tag{38}
$$

为什么下边界 $$1-\varepsilon$$ 消失了？如果 $$\rho<1-\varepsilon$$，原始项 $$\rho A$$ 比裁剪项 $$(1-\varepsilon)A$$ 更小，外层 $$\min$$ 仍然选原始项。也就是说，好动作若被错误地降低了概率，梯度不会被裁掉；它仍有动力恢复。

当好动作的概率已经升到旧概率的 $$1+\varepsilon$$ 倍以上时，继续提升这个样本的概率不再增加该项的目标值。

### 8.2 负优势：只给向下抑制设置平台

当 $$A<0$$ 时，乘负数会翻转不等号，因此

$$
f_A(\rho)=A\max(\rho,1-\varepsilon)
=
\begin{cases}
(1-\varepsilon)A,&\rho<1-\varepsilon,\\
\rho A,&\rho\ge1-\varepsilon.
\end{cases}
\tag{39}
$$

如果坏动作的概率错误地变得更高，哪怕 $$\rho>1+\varepsilon$$，原始项 $$\rho A$$ 也更负，$$\min$$ 会保留它的梯度。只有在坏动作的概率已经被压低到下界以外时，继续压低的额外激励才被截断。

当 $$A=0$$ 时，该样本项恒为零。在两个折点处函数不可微，可使用实现中的次梯度约定；这不影响两侧区域的结论。

### 8.3 梯度写出来就没有歧义

对未处于平台区的样本，由式 (31) 得到

$$
\nabla_\theta f_A(\rho_t)
=A\rho_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\tag{40}
$$

平台区中，该样本的这一项对策略参数的梯度为零。准确说法是“该样本项不再提供继续沿有利方向越界的激励”，而不是“模型参数停止更新”，因为同一批次还有其他 token、其他样本以及可能的辅助项，共享参数仍然会变化。

### 8.4 六种情况的数值检查

设 $$\varepsilon=0.2$$，旧动作概率为 $$0.2$$，比较新概率 $$0.1,0.22,0.3$$，对应比率为 $$0.5,1.1,1.5$$。

| 优势 $$A$$ | 比率 $$\rho$$ | 原始项 $$\rho A$$ | 裁剪项 | 最终较小值 | 对 $$\rho$$ 的斜率 |
|---:|---:|---:|---:|---:|---:|
| $$+2$$ | $$0.5$$ | $$1$$ | $$1.6$$ | $$1$$ | $$+2$$ |
| $$+2$$ | $$1.1$$ | $$2.2$$ | $$2.2$$ | $$2.2$$ | $$+2$$ |
| $$+2$$ | $$1.5$$ | $$3$$ | $$2.4$$ | $$2.4$$ | $$0$$ |
| $$-2$$ | $$0.5$$ | $$-1$$ | $$-1.6$$ | $$-1.6$$ | $$0$$ |
| $$-2$$ | $$1.1$$ | $$-2.2$$ | $$-2.2$$ | $$-2.2$$ | $$-2$$ |
| $$-2$$ | $$1.5$$ | $$-3$$ | $$-2.4$$ | $$-3$$ | $$-2$$ |

最容易误判的是最后一行：坏动作被提高到原来的 $$1.5$$ 倍时，目标仍保留斜率 $$-2$$，没有因为“超过裁剪区间”而失去纠正能力。

### 8.5 Clip 不是硬性信赖域

式 (37) 并没有施加

$$
1-\varepsilon\le\rho_t(\theta)\le1+\varepsilon
$$

这样的可行域约束。概率比可以越界，参数一步也可能跨过平台起点；其他样本的梯度还可能继续改变已处于平台上的 token。更何况，采样动作的比率无法直接限制所有未采样动作或未访问状态的分布变化。

所以需要区分三件事：

1. 裁剪目标逐样本不大于未裁剪 surrogate，这是 $$\min$$ 的代数事实。
2. 裁剪目标不是对真实回报 $$J(\theta)$$ 的普遍严格下界。
3. PPO-Clip 不保证所有概率比都在区间内，也不自动保证 KL 小于某个阈值。

实际更新仍依赖学习率、优化轮数、样本质量、优势估计与 KL 监测等设置。裁剪减轻重复利用旧样本时的过度更新倾向，但不能把任意陈旧数据变成精确的 on-policy 数据。

### 8.6 把公式接成一次完整的 PPO 更新

至此可以把 actor、critic、rollout 和 GAE 连成一个闭环。设 actor 参数为 $$\theta$$，critic 参数为 $$\psi$$，本轮开始时分别保存 $$\theta_{\mathrm{old}}$$ 和 $$\psi_{\mathrm{old}}$$。这里的“保存”可以只是缓存该批轨迹需要的旧 log probability 与旧价值，不一定需要保留另外一份完整网络。

**第一步：用旧策略采样。** 从提示词分布中抽取 $$B$$ 个提示词，让 $$\pi_{\mathrm{old}}$$ 生成回答。记录每个有效位置的状态、动作、是否终止、旧 log probability

$$
\ell_t^{\mathrm{old}}=\log\pi_{\mathrm{old}}(a_t\mid s_t)
$$

和旧价值 $$v_t^{\mathrm{old}}=V_{\psi_{\mathrm{old}}}(s_t)$$。这些数据来自同一批实际生成行为。提示词部分、padding 和 EOS 后的填充位置不能冒充生成动作参与 actor 平均。

**第二步：得到奖励序列。** 用任务评分器、验证器或奖励模型评价回答，把序列评分放在规定的奖励位置。若训练目标包含参考 KL 整形，则按该目标形成整形后的 $$\widetilde r_t$$。接下来的 critic 与 GAE 应使用同一套奖励约定，不能让 actor 学的是包含正则的回报、critic 却无说明地预测另一种回报。

**第三步：计算固定的优势和价值目标。** 例如取 $$\gamma=1$$，对每个回合反向计算

$$
\delta_t=\widetilde r_t+v_{t+1}^{\mathrm{old}}-v_t^{\mathrm{old}},
\qquad
\widehat A_t=\delta_t+\lambda\widehat A_{t+1},
$$

$$
\widehat R_t=\widehat A_t+v_t^{\mathrm{old}}.
$$

真正终止处使用 $$v_T^{\mathrm{old}}=0$$ 和 $$\widehat A_T=0$$；仅被采样窗口截断时按任务定义处理自举。优势和目标随后都保存为不参与梯度传播的数值。若对 actor 的优势做中心化或标准化，应先用原始 GAE 计算 $$\widehat R_t$$，避免把归一化后的相对权重误当作 critic 回报目标。

**第四步：在该批数据上做若干轮小批量更新。** 对每个 minibatch，重新计算当前 $$\log\pi_\theta(a_t\mid s_t)$$，从固定分母得到

$$
\rho_t(\theta)=\exp\bigl(\log\pi_\theta(a_t\mid s_t)-\ell_t^{\mathrm{old}}\bigr),
$$

再按式 (37) 更新 actor。优势 $$\widehat A_t$$ 在这里是停止梯度的权重：不把 actor 项的导数传入优势计算、旧价值或奖励模型。

critic 则拟合固定目标：

$$
L_V(\psi)=\frac{1}{2N_{\mathrm{mb}}}
\sum_{t\in\mathrm{minibatch}}
\bigl(V_\psi(s_t)-\operatorname{stopgrad}(\widehat R_t)\bigr)^2,
\tag{41}
$$

其中 $$N_{\mathrm{mb}}$$ 是本小批量的有效位置数。均方误差的梯度是

$$
\nabla_\psi L_V
=\frac{1}{N_{\mathrm{mb}}}\sum_t
\bigl(V_\psi(s_t)-\widehat R_t\bigr)\nabla_\psi V_\psi(s_t).
$$

因此预测高于目标时会被向下修正，低于目标时会被向上修正。actor 和 critic 可以使用不同网络，也可以共享部分参数；共享参数时，价值拟合也会影响共享表示，但这并不意味着应该沿 actor 的优势权重反向传播。

**可选的熵项。** 一些实现给 actor 添加正系数 $$c_H$$ 的策略熵奖励：

$$
\mathcal H\bigl(\pi_\theta(\cdot\mid s)\bigr)
=-\sum_a\pi_\theta(a\mid s)\log\pi_\theta(a\mid s),
$$

$$
L_{\mathrm{actor}}(\theta)
=L^{\mathrm{clip}}(\theta)
+c_H\widehat{\mathbb E}_t[\mathcal H(\pi_\theta(\cdot\mid s_t))].
\tag{42}
$$

熵衡量同一状态下动作分布的不确定性。正系数通常鼓励保留更多动作概率质量，减缓分布过早变尖；它是可选项，不是 PPO-Clip 定义中不可缺少的一部分，也不能与参考 KL 混为一谈。

**第五步：结束旧批次，重新采样。** 达到规定的优化轮数，或监测到本轮旧策略与新策略的 KL 过大而触发提前停止后，丢弃或按明确的数据策略处理这批 rollout。令下一轮采样策略为更新后的 actor，再重复上述过程。

一批内的多个 minibatch 更新共享同一个生成时基准；新一批重新采样才更新这个基准。有限轮重复使用样本提高了数据利用率，而 PPO 的裁剪和 KL 监测用于减轻这种重复使用带来的过度更新风险。它们并未消除优势估计误差、奖励误差或分布偏移。

## 9. 旧策略约束与参考模型 KL 是两个不同层次

在 RLHF 中需要同时区分

$$
\pi_\theta,\qquad \pi_{\mathrm{old}},\qquad \pi_{\mathrm{ref}}.
$$

- $$\pi_\theta$$：当前正在更新的策略。
- $$\pi_{\mathrm{old}}$$：本批 rollout 实际使用的策略快照，通常在下一轮采样时更新，用于概率比与本次策略变化度量。
- $$\pi_{\mathrm{ref}}$$：作为行为锚点的参考策略，常为 RL 开始前的 SFT 模型，通常长期固定，用于控制相对于这个锚点的偏移。

这三个对象在某个初始时刻可以数值相同，但它们在算法中的职责不同。

PPO 的比率是

$$
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)},
$$

参考正则通常约束的是 $$D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$$。前者服务于采样批次内的局部更新，后者服务于整体训练过程中的行为锚定。PPO 的 $$\varepsilon$$ 与参考 KL 的 $$\beta$$ 因而不能互换：多轮很小的 PPO 更新仍可能逐步远离参考模型。



## 10. KL 正则项：序列目标怎样落实到 token

### 10.1 把序列 KL 展开为条件 KL

对固定输入 $$x$$，根据自回归分解，

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=\sum_{t=0}^{T-1}\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{ref}}(a_t\mid s_t)}.
$$

对当前策略生成的完整序列取期望，再按前缀条件化，得到 KL 的链式分解：

$$
D_{\mathrm{KL}}(\pi_\theta(y\mid x)\Vert\pi_{\mathrm{ref}}(y\mid x))
=\mathbb E_{y\sim\pi_\theta}\sum_{t=0}^{T-1}
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid s_t)\Vert\pi_{\mathrm{ref}}(\cdot\mid s_t)\right).
$$

右侧的前缀也来自当前策略。只在固定旧前缀上计算条件 KL，得到的是一个局部正则项，不能忽略前缀分布的区别而直接称为完整序列 KL。

单个已采样 token 的对数比率

$$
k_t=\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{ref}}(a_t\mid s_t)
$$

可以为负，但它在 $$a_t\sim\pi_\theta(\cdot\mid s_t)$$ 下的条件期望是非负的 KL。KL 非负不代表每个 Monte Carlo 样本非负。

### 10.2 奖励塑形为什么能包含一个依赖参数的项

设固定输入下 $$p_\theta(y)=\pi_\theta(y\mid x)$$，$$q(y)=\pi_{\mathrm{ref}}(y\mid x)$$。直接微分，而不是先把 KL 当成常数：

$$
\begin{aligned}
\nabla_\theta J_\beta
&=\sum_y\nabla_\theta p_\theta(y)
\left[R(y)-\beta\log\frac{p_\theta(y)}{q(y)}\right]
-\beta\sum_y p_\theta(y)\nabla_\theta\log p_\theta(y)\\
&=\mathbb E_{p_\theta}\left[
\left(R(y)-\beta\log\frac{p_\theta(y)}{q(y)}\right)
\nabla_\theta\log p_\theta(y)\right].
\end{aligned}
$$

第二行成立，因为最后一项等于 $$-\beta\nabla_\theta\sum_y p_\theta(y)=0$$。这说明，在**当前策略采样、同一点求导**的条件下，可以把采样出的对数比率纳入权重，再停止该权重的梯度，得到正确的期望梯度。

若只有终局任务奖励，一个常见的逐步塑形写法是

$$
\widetilde r_t
=-\beta\left[\log\pi_{\mathrm{old}}(a_t\mid s_t)-\log\pi_{\mathrm{ref}}(a_t\mid s_t)\right]
+\mathbf 1[t=T-1]R(x,y).
$$

采样时计算并固定这些奖励，随后用它们构造 return、价值目标和 GAE。这在 $$\theta=\theta_{\mathrm{old}}$$ 处与上述推导衔接；对同一批数据做多轮 PPO 更新时，它已经成为局部代理训练过程，不能声称每一步都是当前全局正则目标的精确梯度。

另一些实现把 KL 作为显式损失项，而不是全部并入奖励。两种写法需要连同采样分布、估计器和梯度路径一起比较，不能只看有没有一个 $$\beta D_{\mathrm{KL}}$$ 符号。若同时使用，要明确是否有意施加两次不同的正则。



## 11. GRPO 及其变体：用同题回答构造学习信号

本节令回答下标从 1 开始，$$y_i=(y_{i,1},\ldots,y_{i,L_i})$$，长度记为 $$L_i$$；$$G$$ 是同一输入的回答组大小，不要与前文回报 $$G_t$$ 混淆。

### 11.1 把 critic 换成同题回答的比较

#### 读公式前：本节的记号

- $$\mu$$ 是训练问题的分布；$$x\sim\mu$$ 表示先随机选一道题。
- $$G$$ 是每题生成的回答数量，$$i\in\{1,\ldots,G\}$$ 是回答索引；$$L_i$$ 是第 $$i$$ 条回答的有效 token 数，$$t$$ 是 token 位置，$$y_{i,<t}$$ 是该回答前 $$t$$ 个位置之前的前缀。
- $$\theta$$ 是待训练模型的参数；$$\pi_\theta$$ 是当前策略，即模型给文本的概率分布；$$\pi_{\mathrm{old}}$$ 是采样时冻结的行为策略，$$\pi_{\mathrm{ref}}$$ 是用于正则化的参考策略。后两者在一次策略更新中不求梯度，且用途不同。
- $$R(x,y)$$ 是给完整回答的终局分数。在奖励梯度的基础推导中，假设它本身不显式依赖 $$\theta$$；带 KL 的参数相关奖励会单独说明。
- $$\mathbb E$$ 表示对下标指定的随机采样取平均。$$\mathbb E[\cdot\mid x]$$ 表示固定问题，只对回答采样取平均；$$\nabla_\theta$$ 表示对模型参数求梯度。
- i.i.d. 意为给定同一道题，每条回答独立地从同一分布采样。rollout 指用冻结模型生成这批回答。stop-gradient/detach 指使用一个量的数值，但不沿它反向求导。

#### 用一组回答构造优势

常见 PPO 实现中的 critic 试图根据前缀 $$s_t=(x,y_{<t})$$ 预测后续回报，再用 GAE 等方法估计每个 token 的优势。GRPO 的基本想法是：对同一道题采样多条答案，用它们的奖励作比较，避免训练额外的价值模型。DeepSeekMath 于 2024 年 2 月 5 日首次提出 GRPO；这里采用其 v3 的 outcome-supervision 定义。[DeepSeekMath，§4.1](https://arxiv.org/html/2402.03300v3)

默认 $$G\ge2$$。对每个 $$x$$，从行为策略独立采样：

$$
y_1,\ldots,y_G\overset{\mathrm{i.i.d.}}{\sim}\pi_{\mathrm{old}}(\cdot\mid x),
\qquad R_i=R(x,y_i).
$$

计算组内均值、标准差和标准化优势：

$$
\bar R=\frac1G\sum_{j=1}^G R_j,\qquad
s_R=\sqrt{\frac1G\sum_{j=1}^G(R_j-\bar R)^2},\qquad
\widehat A_i=\frac{R_i-\bar R}{s_R+\epsilon_{\mathrm{num}}}.
$$

这里用总体形式的组内标准差，即分母为 $$G$$。若实现使用 $$G-1$$，忽略数值稳定项且方差非零时会再差一个固定因子；数值稳定项应与 PPO 的裁剪阈值分开命名。以上优势是在 rollout 后计算并停止梯度的训练标签。

令

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}
     {\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})},
$$

$$
\ell_{\mathrm{clip}}(\rho,A)
=\min\!\left\{\rho A,\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A\right\}.
$$

其中 $$\rho$$ 是概率比，$$A$$ 是已经算好的优势；$$\operatorname{clip}(z,l,u)=\min(\max(z,l),u)$$ 把数值截到区间 $$[l,u]$$。$$\varepsilon>0$$ 控制裁剪范围；$$\beta\ge0$$ 是 KL 正则系数，$$\widehat k_{i,t}$$ 是该 token 的 KL 估计项，具体定义和梯度陷阱在后文展开。

原始 outcome-GRPO 的结构为

$$
J_{\mathrm{GRPO}}(\theta)=
\mathbb E_{x\sim\mu,\,\{y_i\}_{i=1}^G\sim\pi_{\mathrm{old}}(\cdot\mid x)}
\left[\frac1G\sum_{i=1}^G\frac1{L_i}\sum_{t=1}^{L_i}
\left\{\ell_{\mathrm{clip}}(\rho_{i,t}(\theta),\widehat A_i)
-\beta\widehat k_{i,t}(\theta)\right\}\right].
$$

终局奖励的同一个 $$\widehat A_i$$ 被广播到该回答所有 token。它没有突然获得“哪一步推理真正贡献了成功”的精确识别能力；其信用分配仍是整条轨迹回报乘各 token 的 score。相对于完整 PPO 训练栈，主要变化是：移除价值网络与价值损失，替换优势估计器，选择组内归一化与回答长度归一化。裁剪 surrogate 和旧策略 rollout 仍然保留。

**省掉 critic，不等于省掉所有额外计算。** 组内估计需要同一问题的多个样本，采样成本、组内差异性和有效奖励信号成为新的关键因素。也不要把 GRPO 与“只用规则奖励”画等号：优势估计方法与奖励来源是两条不同的设计轴。

#### 手算一组：从奖励到真正的更新系数

假设同一道题采样四条答案，奖励为 $$(1,0,1,0)$$。组均值为 $$0.5$$，组标准差为 $$0.5$$，所以忽略极小的数值稳定项后：

$$
R_i-\bar R=(0.5,-0.5,0.5,-0.5),\qquad
\widehat A_i=(1,-1,1,-1).
$$

正优势意味着“提高这条回答中已采样 token 的概率”，负优势意味着“降低它们的概率”，但幅度还受概率比、裁剪和 reduction 共同影响。

取 $$\varepsilon=0.2$$。某个 token 在旧策略中的概率是 $$0.2$$，当前变成 $$0.28$$，于是 $$\rho=1.4$$：

- 若它属于优势 $$+1$$ 的回答，surrogate 为 $$\min(1.4,1.2)=1.2$$，已经进入有利方向的平台区；仅这个样本项不再奖励进一步增大概率。
- 若它属于优势 $$-1$$ 的回答，surrogate 为 $$\min(-1.4,-1.2)=-1.4$$，仍有梯度去阻止这种有害的概率增加。

如果负优势 token 的当前概率降至 $$0.14$$，则 $$\rho=0.7$$，其 surrogate 为 $$\min(-0.7,-0.8)=-0.8$$，在有利方向进入平台区。裁剪具有这种与优势符号相关的不对称性。整个网络共享参数，因此单个项平台化不意味着该 token 概率在之后绝对不会改变。

### 11.2 一个容易被略过的细节：组均值里包含样本自己

先去掉标准差、长度归一化、裁剪和 KL，并在真正 on-policy 的点 $$\pi_\theta=\pi_{\mathrm{old}}$$ 分析奖励梯度。固定问题 $$x$$，定义

$$
S_i=\nabla_\theta\log\pi_\theta(y_i\mid x)
=\sum_{t=1}^{L_i}\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}),
\qquad
g(x)=\mathbb E[R_iS_i\mid x].
$$

这里的 score 指对数概率的参数梯度，不是奖励分数。在支持集不随参数变化、完整轨迹概率归一化等常规正则条件下，score 的期望为零：

$$
\mathbb E[S_i\mid x]
=\sum_y\pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x)
=\nabla_\theta\sum_y\pi_\theta(y\mid x)=0.
$$

如果 baseline 是只依赖问题的 $$b(x)$$，便有

$$
\mathbb E[(R_i-b(x))S_i\mid x]=g(x).
$$

但组均值 $$\bar R$$ 包含当前样本的 $$R_i$$。因此

$$
\begin{aligned}
\mathbb E[\bar R S_i\mid x]
&=\frac1G\mathbb E[R_iS_i\mid x]
 +\frac1G\sum_{j\ne i}\mathbb E[R_jS_i\mid x]\\
&=\frac1G g(x),
\end{aligned}
$$

其中交叉项为零，依赖的是“给定 $$x$$，不同回答独立”的假设。故

$$
\boxed{
\mathbb E\!\left[\frac1G\sum_{i=1}^G(R_i-\bar R)S_i\,\middle|\,x\right]
=\frac{G-1}{G}g(x).
}
$$

这不是严格等于原奖励梯度，而是少了一个 $$1/G$$ 的比例。固定 $$G$$ 时，它没有改变这一简化估计器的期望方向；$$G=2$$ 时系数是 $$1/2$$，$$G=8$$ 时是 $$7/8$$。但如果组大小会变、KL 或其他损失没有一起缩放，这个系数就不仅是记号问题。即使 $$G$$ 固定，“吸收到学习率”也只是简单梯度更新下的解释，不能无条件断言带 Adam、梯度裁剪、权重衰减的整个训练轨迹完全相同。

#### RLOO 恰好去掉这个自包含项

定义 leave-one-out baseline：

$$
b_{-i}=\frac1{G-1}\sum_{j\ne i}R_j,
\qquad
A_i^{\mathrm{LOO}}=R_i-b_{-i}.
$$

给定问题，$$b_{-i}$$ 与 $$y_i$$ 独立，于是

$$
\mathbb E[A_i^{\mathrm{LOO}}S_i\mid x]=g(x).
$$

而且存在逐样本恒等式：

$$
\boxed{A_i^{\mathrm{LOO}}=\frac{G}{G-1}(R_i-\bar R).}
$$

RLOO 在 RLHF 中的系统研究见 2024 年 2 月 22 日的《Back to Basics》；论文将其作为早已有之的 REINFORCE 多样本估计器用于语言模型，因此不宜说该文首次发明了 leave-one-out baseline。[RLOO，§2.3](https://arxiv.org/html/2402.14740v2)

这里“无偏”的对象是理想 on-policy、未裁剪、没有随机长度/标准差权重的 policy-gradient 估计器。把 RLOO 优势放入多轮 PPO 裁剪更新后，整个算法不会因此自动成为原目标梯度的无偏估计器。

**适用前提很重要。** beam search、对组内样本施加强制去重或相互依赖的采样、以及依赖整组回答共同定义的奖励，都可能破坏上述独立性证明。baseline 本身停止梯度也不能消除统计相关性：detach 解决计算图问题，不能把相关随机变量变成独立随机变量。

### 11.3 除以组标准差，不只是“让训练更稳定”

加入标准差后，估计器变成

$$
\widehat g_{\mathrm{std}}(x)
=\frac1G\sum_i\frac{R_i-\bar R}{s_R+\epsilon_{\mathrm{num}}}S_i.
$$

不能写成

$$
\mathbb E[\widehat g_{\mathrm{std}}]
\stackrel{\text{通常错误}}{=}
\frac{G-1}{G}\frac{\nabla_\theta\mathbb E[R]}{\mathbb E[s_R]}.
$$

原因有两层：

1. 一般而言，随机变量之比的期望不等于期望之比。
2. $$s_R$$ 与组内的奖励以及 score 是相关的，不能从期望中抽出来。

若先作大 $$G$$ 的近似，在当前策略附近把真实奖励标准差 $$\sigma_R(x)$$ 当作本轮固定量，则会看到一种按问题加权的方向：

$$
\widehat g_{\mathrm{std}}(x)\approx
\frac1{\sigma_R(x)}\nabla_\theta\mathbb E[R\mid x].
$$

因此不同问题会得到不同的相对梯度权重。由于这些权重又随策略改变，这个更新向量场不应不加证明就叫作“某个固定加权奖励目标的精确梯度”。

#### 二元奖励为什么会突出组内少数派？

若 $$R_i\in\{0,1\}$$，组内成功比例为 $$\widehat p=k/G$$，则在 $$0<k<G$$ 时

$$
s_R=\sqrt{\widehat p(1-\widehat p)},
$$

$$
\widehat A_{\mathrm{success}}
=\sqrt{\frac{1-\widehat p}{\widehat p}},
\qquad
\widehat A_{\mathrm{failure}}
=-\sqrt{\frac{\widehat p}{1-\widehat p}}.
$$

例如 $$G=8$$：

- 只有 1 个成功样本：成功者优势为 $$\sqrt7$$，每个失败者为 $$-1/\sqrt7$$。
- 有 7 个成功样本：每个成功者为 $$1/\sqrt7$$，唯一失败者为 $$-\sqrt7$$。
- 恰好 4 成 4 败：优势分别为 $$+1,-1$$。

这说明标准化加强了组内少数派样本的系数。它不等于“所有难题都会获得很大有效梯度”：完全失败的组没有组内差异，仍然没有这部分学习信号；梯度方向、出现概率和 score 大小也不能仅由优势系数判断。

#### 一个反例：标准化确实可能改变方向

考虑单步三动作 softmax 策略，三个动作概率都为 $$1/3$$，取 $$G=2$$，忽略数值稳定项。两个奖励不同时，其标准化优势永远是 $$+1,-1$$，与奖励差距大小无关；两个奖励相同时置零。

因此，奖励向量 $$r=(0,1,10)$$ 与 $$r'=(0,9,10)$$ 会产生相同的期望标准化组梯度。枚举九个有序采样对，可得对三个 logits 的梯度都是

$$
\mathbb E[\widehat g_{\mathrm{std}}]
=\left(-\frac29,0,\frac29\right).
$$

但真实期望奖励梯度是 $$p_j(r_j-\sum_k p_kr_k)$$，分别为

$$
\nabla_\theta\mathbb E[r]
=\left(-\frac{11}{9},-\frac89,\frac{19}{9}\right),
\quad
\nabla_\theta\mathbb E[r']
=\left(-\frac{19}{9},\frac89,\frac{11}{9}\right).
$$

中间动作应该下降还是上升，本来取决于奖励的具体数值；这个小组标准化估计器却给它零期望更新。由此可见，标准化造成的差异不能总用一个全局学习率解释。

### 11.4 长度归一化实际上在给谁加权？

令 $$f_{i,t}$$ 表示一个 token 的 surrogate 项。三种常见 reduction 分别是：

$$
J_{\mathrm{sample}}
=\frac1G\sum_i\frac1{L_i}\sum_t f_{i,t},
$$

$$
J_{\mathrm{token}}
=\frac{\sum_i\sum_t f_{i,t}}{\sum_i L_i},
$$

$$
J_{\mathrm{constant}}
=\frac1{GC}\sum_i\sum_t f_{i,t},
\qquad C>0\text{ 为固定常数}.
$$

它们并非等价的编码风格：

- 第一种先对每条回答取 token 平均，再对回答平均；每条回答在 scalar loss 聚合上权重相同，每个 token 的权重是 $$1/(GL_i)$$。
- 第二种对所有有效 token 共同取平均；同一批次中每个 token 权重相同，长回答拥有更多项。
- 第三种保留 token 求和，只用固定数值尺度控制梯度大小；不会根据本次样本长度改变归一化因子。

假设两条回答长度分别为 10 和 100，优势都为 $$+1$$，并在 $$\rho=1$$ 的点查看每个 token score 前的系数：第一种分别是 $$1/20$$ 和 $$1/200$$，相差 10 倍；第二种都为 $$1/110$$；第三种若 $$C=100$$，都为 $$1/200$$。如果优势是 $$-1$$，相同系数变为负值。这解释了为何回答平均会对长回答中的单个错误 token 给出较弱的惩罚。

不过，“token 系数大十倍”不等于“完整参数梯度范数一定大十倍”。不同 token 的 score 向量可能同向、抵消，或者集中在不同参数上。不要从 loss 权重直接推导确定的生成长度规律。

#### 为什么除以 $$L_i$$ 会破坏朴素 baseline 论证？

完整序列的奖励梯度包含的是

$$
\mathbb E\!\left[R\sum_t\nabla\log\pi_\theta(y_t\mid s_t)\right]
=\mathbb E[RS].
$$

改成回答内平均后得到 $$\mathbb E[(R-b)S/L]$$。固定问题 $$x$$，并先把 $$b=b(x)$$ 视为本轮更新中的固定量，通常仍有

$$
\mathbb E[bS/L]=b\nabla_\theta\mathbb E[1/L]\ne0.
$$

因为 $$L=L(y)$$ 是动作序列本身的属性，$$b/L(y)$$ 已经不是与动作无关的 baseline。对固定的 $$b$$，这个局部更新相当于

$$
\nabla_\theta\mathbb E[(R-b)/L],
$$

而非原本的 $$\nabla_\theta\mathbb E[R]$$。所以“只是做了平均，目标没变”的说法尤其危险。

第二种 token mean 也需要谨慎：在一个已经采完、固定的 batch 内，它与常数归一化仅差共同倍数；但跨 batch，分母 $$\sum_iL_i$$ 是随机变量，与分子相关。因此不能据此声称其期望严格等于原始奖励梯度的固定倍数。跨 GPU、microbatch 或梯度累积时，若各分块先各自取 mean，再不加权平均，甚至未必等价于整个 global batch 的 token mean。

### 11.5 Dr. GRPO：去掉两种归一化，但别扩大“无偏”的含义

Dr. GRPO 首发于 2025 年 3 月 26 日，核心做法是去掉回答长度除数和组内标准差除数，保留组均值中心化，实际实现可用固定生成预算作数值归一化。2025 年 10 月修订版附录明确写出了组中心化与 RLOO 的常数倍关系。[Dr. GRPO，§3.2 与附录 A](https://arxiv.org/html/2503.20783v2)

用统一记号写，奖励部分为

$$
J_{\mathrm{Dr}}(\theta)
=\mathbb E\!\left[
\frac1{GC}\sum_i\sum_{t=1}^{L_i}
\ell_{\mathrm{clip}}(\rho_{i,t}(\theta),R_i-\bar R)
\right].
$$

理解它时应分成三个层次：

1. 去掉 $$1/L_i$$ 和 $$1/s_R$$，消除了这两项引入的样本/问题随机加权。
2. 未裁剪、on-policy、固定 $$G$$ 时，含自身的组均值仍使期望奖励梯度缩放为 $$(G-1)/(GC)$$；采用 RLOO 或乘 $$G/(G-1)$$ 可消除自包含基线带来的缩放；若要数值上无偏估计原始奖励梯度，还需去掉 $$1/C$$（或再乘 $$C$$）。
3. 保留 PPO clipping、复用旧策略数据后，优化的依然是近端 surrogate；这不是“整个算法在所有参数位置都是原目标的无偏梯度”。

也不要反过来把标准差归一化一律称作错误。若有意希望不同问题的奖励尺度更可比，它可以是明确的算法选择。真正需要避免的是：修改了权重以后，仍使用原始未加权目标的证明来担保它。

### 11.6 零方差组与 DAPO：有信号的 batch 是怎样产生的？

若一组奖励完全相同，则 $$R_i-\bar R=0$$。安全实现应令整个组的优势为零或过滤该组，不能让 $$0/0$$ 产生 NaN。加上 $$\epsilon_{\mathrm{num}}$$ 只解决数值问题，并不会凭空创造奖励信号。$$G=1$$ 时，组中心化奖励恒为零，标准 RLOO 也无定义。

对成功率为 $$p(x)$$ 的二元奖励，独立采样 $$G$$ 次得到至少一对成功/失败的概率是

$$
P_{\mathrm{mix}}(x)=1-p(x)^G-(1-p(x))^G.
$$

例如 $$G=8,p=0.99$$ 时，这个概率约为 $$0.0773$$：超过九成组对组相对奖励学习没有信号。$$p=0.5$$ 时则约为 $$0.9922$$。组内全错既可能说明题太难，也可能只是当前这组抽样没有成功，不能直接断言模型的真实成功率为零。

DAPO 于 2025 年 3 月 18 日提出，包含四项主要设计：上下裁剪阈值解耦、动态补采非退化组、token-level loss 聚合、以及超长回答的奖励处理；其报告中的训练配置移除了 KL 正则项。[DAPO，§2.3、§3](https://arxiv.org/html/2503.14476v2)

#### 解耦裁剪边界

$$
\ell_{\mathrm{DAPO}}(\rho,A)
=\min\!\left\{\rho A,
\operatorname{clip}(\rho,1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}})A
\right\}.
$$

增加上界允许正优势动作获得更大的有利概率比更新；下界可以保持不变。其裁剪仍是 surrogate 的平台区，而不是将实际策略硬性限制在某个概率区间，更不保证所有 token 的 KL 都受到某个确定上界约束。

#### 动态采样的统计含义

过滤零优势组并补采，可以让每次优化使用更多有奖励信号的组。但“改变了抽样分布”不自动等于“改变了期望方向”，需要看估计器。

一个可精确分析的简化情形是：每次重新抽问题和整组回答；只在所有梯度项都为零时丢弃；对接受的组作等权平均。令 $$Z$$ 表示“保留”，$$g_{\mathrm{group}}=0$$ 对所有 $$Z=0$$ 的组成立，则

$$
\mathbb E[g_{\mathrm{group}}\mid Z=1]
=\frac{\mathbb E[g_{\mathrm{group}}]}{P(Z=1)}.
$$

此时仅有一个全局比例变化。反之，若固定每个问题反复采到一组混合结果，再让每个问题等权，则不同问题出现 $$1/P_{\mathrm{mix}}(x)$$ 的不同因子。若还有 KL/熵梯度、随机 token 分母、额外奖励、截断 mask，前面的“被丢弃组梯度全为零”及简单等权前提也要重新检查。因此应根据实际采样与 reduction 流程分析，不能一句“动态采样一定无偏”或“一定改变难度权重”概括所有实现。

#### 超长回答必须区分“错”和“没来得及答完”

若生成被长度上限截断，没有看到正确最终答案，并不证明已有推理过程错误。过滤截断回答的训练项、设置渐进长度惩罚、或给终止状态明确语义，是不同的设计选择。长度惩罚若直接进入奖励，目标本身就已改变为“正确性与成本的折中”。这些工程选择应和模型的自然 EOS 结束分开记录。

### 11.7 GSPO：整条回答一起裁剪，但这个比率不是完整 IS

GSPO 于 2025 年 7 月 24 日首次提交，将共享的序列级比率用于整条回答的裁剪。其定义采用**长度归一化的序列似然比**。[GSPO，§4](https://arxiv.org/html/2507.18071v2)

真正的轨迹重要性采样比率为

$$
W_i(\theta)=\frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}
=\prod_{t=1}^{L_i}\rho_{i,t}(\theta).
$$

GSPO 使用的是

$$
q_i(\theta)=W_i(\theta)^{1/L_i}
=\exp\!\left(\frac1{L_i}\sum_t\log\rho_{i,t}(\theta)\right),
$$

即 token 比率的几何平均。其 surrogate 为

$$
J_{\mathrm{GSPO}}(\theta)
=\mathbb E\left[\frac1G\sum_i
\ell_{\mathrm{clip}}(q_i(\theta),\widehat A_i)\right].
$$

在未裁剪区域内直接求导：

$$
\nabla_\theta q_i
=\frac{q_i}{L_i}\sum_t\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t}),
$$

$$
\nabla_\theta J_{\mathrm{GSPO}}
=\mathbb E\left[\frac1G\sum_i
\frac{q_i\widehat A_i}{L_i}\sum_t\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})\right].
$$

与原始 GRPO 相比，同一条回答中的 token 共享 $$q_i$$ 和序列级裁剪开关，不再各用 $$\rho_{i,t}$$。但是 $$1/L_i$$ 仍然存在于梯度中，只是来自指数里的长度归一化。它不是 Dr. GRPO 那种去除长度归一化的设计。

还应划清三条边界：

- 一般来说 $$q_i\ne W_i$$，所以 $$\mathbb E_{\mathrm{old}}[q_if(y_i)]\ne\mathbb E_\theta[f(y_i)]$$。不能把稳定数值范围的几何平均比率称作原轨迹分布的精确 IS 校正。
- 即使一条回答很长、整体 $$W_i$$ 很大，$$q_i$$ 仍可能接近 1。例如每个 token 比率均为 1.01，1000 个 token 的 $$W_i=1.01^{1000}\approx 2.1\times10^4$$，而 $$q_i=1.01$$。这是两种尺度，不是严格的整体似然约束。
- “每个条件分布只抽一个 token，所以 importance sampling 数学上不成立”不是正确的普遍结论。单样本 IS 可以无偏，只是方差很大。token 比率可以校正固定前缀下的条件动作期望；它不能单独修正前缀分布、后续轨迹分布和整条序列的分布差异。PPO 使用的是局部 surrogate，而非声称每次更新都完成了精确的完整轨迹换测度。

这也是阅读算法论文时应保持的区分：作者的训练稳定性观察是经验事实的报告；对 IS 或全局收敛的数学解释，还需要逐条检查假设。

### 11.8 补充：KL 估计值与 KL 梯度必须分别检查

原始 GRPO 使用如下非负 KL 估计项。为避免符号混淆，这里固定一个前缀 $$s$$，令 $$p(a)=\pi_{\mathrm{ref}}(a\mid s)$$、$$q_\theta(a)=\pi_\theta(a\mid s)$$，并定义

$$
u(a)=\frac{p(a)}{q_\theta(a)},\qquad
k_3(a)=u(a)-\log u(a)-1.
$$

由 $$u-1\ge\log u$$，逐样本 $$k_3\ge0$$。在 $$p,q_\theta$$ 具有共同、固定且各点概率严格为正的支持集，并且 $$a\sim q_\theta$$ 时：

$$
\mathbb E_{q_\theta}[k_3]
=\underbrace{\mathbb E_{q_\theta}[p/q_\theta]}_{=1}
 +\mathbb E_{q_\theta}[\log(q_\theta/p)]-1
=D_{\mathrm{KL}}(q_\theta\|p).
$$

这是**数值期望**的恒等式。原始估计形式见 [DeepSeekMath，式 $$4$$](https://arxiv.org/html/2402.03300v3)；下面继续对梯度作独立核算。

如果先采样、把离散动作冻结，然后只反向传播 $$k_3$$，得到

$$
\nabla_\theta k_3(a)
=(1-u(a))\nabla_\theta\log q_\theta(a).
$$

对当前策略采样取期望：

$$
\mathbb E_{q_\theta}[\nabla_\theta k_3]
=-\mathbb E_p[\nabla_\theta\log q_\theta]
=\nabla_\theta D_{\mathrm{KL}}(p\|q_\theta),
$$

它一般不是 $$\nabla D_{\mathrm{KL}}(q_\theta\|p)$$。因为求导与取期望交换时，还必须包含采样分布随 $$\theta$$ 变化的 score 项：

$$
\begin{aligned}
\nabla_\theta\mathbb E_{q_\theta}[k_3]
&=\mathbb E_{q_\theta}
\left[k_3\nabla\log q_\theta+\nabla k_3\right]\\
&=\mathbb E_{q_\theta}
\left[\log(q_\theta/p)\nabla\log q_\theta\right].
\end{aligned}
$$

后者才是所需的 reverse-to-reference KL 梯度。这里固定了前缀；如果讨论完整轨迹 KL，还必须考虑前缀状态分布随策略改变，不能把固定状态的推导直接当作完整序列目标的证明。

如果样本来自固定行为分布 $$m$$，且 $$m$$ 覆盖上述共同支持集，精确的固定前缀换测度形式是

$$
D_{\mathrm{KL}}(q_\theta\|p)
=\mathbb E_{a\sim m}\left[\frac{q_\theta(a)}{m(a)}k_3(a)\right].
$$

此时必须对 $$q_\theta/m$$ 也正确求导，才能恢复完整梯度；若 detach 该权重、裁剪该权重、或根本未加该权重，就不能继续使用同一个无偏梯度结论。尤其在同一批旧数据上更新多轮以后，连未经修正的 $$\mathbb E_m[k_3]$$ 数值通常也不再等于当前 KL。

与此对照，若使用完整序列采样的 log-ratio 奖励

$$
\widetilde R_\theta(y)=R(y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},
$$

则在真正 on-policy 时，将它作为停止梯度的 REINFORCE 回报仍可给出 KL 正则化序列目标的正确期望梯度：漏掉的显式导数为 $$-\beta\mathbb E[\nabla\log\pi_\theta(y\mid x)]=0$$。这依赖特定 log-ratio 结构及 on-policy 期望，不能推广成“任意参数相关奖励都可以 detach”。

**阅读代码时必须同时看四件事：样本从哪里来、loss 写了什么、哪些量停止梯度、reduction 怎样做。** 只看到公式中写了 $$D_{\mathrm{KL}}$$，还不足以确定实际优化的梯度。

### 11.9 最小训练流程与排错清单

一个 outcome-GRPO 家族的训练循环可以概括为：

1. 选问题并冻结行为策略快照；用它为每个问题采样 $$G$$ 条回答，保存旧 token log-prob、结束位置和截断标记。若温度、top-p 或其他采样设置改变了行为分布，概率比中的分母与实际采样分布是否匹配也需要核对。
2. 计算奖励，按问题正确分组；计算中心化或 LOO 优势，并明确是否除以标准差。
3. 处理零方差组；如果采用动态采样，记录总采样量、保留比例与接受规则。
4. 用当前策略重算有效回答 token 的 log-prob；以 log-prob 差计算比率，避免把 prompt 或 padding 混进 loss。自然生成的 EOS 是策略动作，通常应属于有效回答；被预算强制截断则是另外的终止机制。
5. 按选定 token/sequence 裁剪方式构造 surrogate；明确是 sample mean、global token mean 还是固定分母。
6. 根据明确的目标和梯度实现添加 KL/熵等项；保持旧策略与参考策略的角色分离。
7. 对这一批数据更新指定轮数，随后重新 rollout；持续监控奖励、独立验证表现、KL、熵、裁剪比例、长度分布、截断率和有效组比例。



## 12. DPO：从同一个正则目标走向偏好拟合

PPO 与 GRPO 都需要生成样本、获得评分，再估计更新方向。DPO 的入口不同：先求 KL 正则问题在分布空间的最优解，然后把该解代回偏好模型。这一节用独立推导说明两条路线的联系，不把 DPO 混称为一种策略梯度估计器。

### 12.1 固定输入，先解最优分布

暂时不考虑神经网络能否表示最优策略。固定输入 $$x$$，令 $$p(y)$$ 为待优化的概率分布，$$q(y)=\pi_{\mathrm{ref}}(y\mid x)$$，假设相关回答上 $$q(y)>0$$ 且归一化常数有限。问题为

$$
\max_p\sum_y p(y)R(x,y)-\beta\sum_y p(y)\log\frac{p(y)}{q(y)},
\qquad\sum_y p(y)=1.
$$

引入乘子 $$\lambda$$：

$$
\mathcal F(p,\lambda)
=\sum_y p(y)R(x,y)-\beta\sum_y p(y)\log\frac{p(y)}{q(y)}
+\lambda\left(\sum_y p(y)-1\right).
$$

对每一个 $$p(y)$$ 求偏导：

$$
\frac{\partial\mathcal F}{\partial p(y)}
=R(x,y)-\beta\left(\log\frac{p(y)}{q(y)}+1\right)+\lambda=0.
$$

整理后，$$p(y)$$ 与 $$q(y)e^{R(x,y)/\beta}$$ 成正比。利用总概率为一，得到

$$
p^*(y)=\frac{q(y)e^{R(x,y)/\beta}}{Z(x)},
\qquad Z(x)=\sum_y q(y)e^{R(x,y)/\beta}.
$$

也可以直接将原目标改写为

$$
\beta\log Z(x)-\beta D_{\mathrm{KL}}(p\Vert p^*),
$$

于是最优性由 KL 非负立即得到。这个变形还说明参考策略的支持集限制：如果 $$q(y)=0$$，有限 KL 的 $$p$$ 不能给这个回答分配正概率。

### 12.2 奖励差中的配分函数消失

将最优解倒过来写：

$$
R(x,y)=\beta\log\frac{p^*(y)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
$$

同一输入下两个回答相减，最后的 $$\beta\log Z(x)$$ 消去：

$$
R(x,y_w)-R(x,y_l)
=\beta\left[
\log\frac{p^*(y_w)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{p^*(y_l)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right].
$$

用参数化策略 $$\pi_\theta$$ 表示这个最优策略，并代回 Bradley–Terry 偏好似然，定义

$$
z_\theta=\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right],
\qquad
\mathcal L_{\mathrm{DPO}}=-\mathbb E_{\mathcal D}\log\sigma(z_\theta).
$$

注意训练期望在固定偏好数据 $$\mathcal D$$ 上，不是新生成的 $$y\sim\pi_\theta$$。又因为

$$
\frac{d}{dz}[-\log\sigma(z)]
=-\frac{\sigma(z)(1-\sigma(z))}{\sigma(z)}
=-\sigma(-z),
$$

一对样本的梯度是

$$
\nabla_\theta\ell_{\mathrm{DPO}}
=-\beta\sigma(-z_\theta)
\left[\nabla_\theta\log\pi_\theta(y_w\mid x)
-\nabla_\theta\log\pi_\theta(y_l\mid x)\right].
$$

若这对偏好尚未被策略很好地区分，$$z_\theta$$ 小，权重大；若已经明显符合偏好，$$z_\theta$$ 大，权重减小。它沿两个完整回答的相对对数概率方向更新，序列概率仍可展开为 token 对数概率的和。

这个推导依赖偏好模型和正则化问题的假设。有限偏好数据、参数化限制、优化误差和分布外行为仍然存在；不能把闭式最优分布推导理解为“有限步 DPO 必然精确等价于任意 PPO 训练”。它给出的是另一种处理偏好对齐问题的建模与优化路径。[DPO 原论文](https://arxiv.org/abs/2305.18290)



## 13. 从公式到实现：一次更新必须对齐的量

上面的公式成立于明确的采样与求导约定。落到训练代码时，以下细节不是附属工程问题，而是决定“到底在实现哪个数学对象”的一部分。

### 13.1 采样概率必须与分母一致

若采样时采用温度 $$\tau$$，实际策略是

$$
\pi_{\theta,\tau}(a\mid s)
=\frac{\exp(z_\theta(s,a)/\tau)}{\sum_b\exp(z_\theta(s,b)/\tau)}.
$$

重要性比率分母需要对应真正产生样本的行为分布。采样用温度后的分布，分母却使用未经温度变换的概率，就不再是前面证明过的那个换测度比率。top-k、top-p 截断会进一步改变支持集；被行为策略置零的动作无法靠有限的比率从旧样本中补回来。

实际大规模系统还可能有采样引擎与训练引擎的数值差异，以及异步更新造成的策略滞后。这些都会让“old”不只是一个变量名。应当明确保存或重建哪一版模型、哪一种采样分布的对数概率。

### 13.2 mask 与 reduction 对应数学求和范围

设 $$m_{it}\in\{0,1\}$$ 标记有效生成 token。长度应当从同一 mask 计算：$$T_i=\sum_t m_{it}$$。padding 不是动作，提示词也不是本轮采样得到的回答动作；EOS 是否计入需要与终止定义一致。

以下两种实现并不相同：

$$
\frac1B\sum_{i=1}^B\frac1{T_i}\sum_t m_{it}\ell_{it},
\qquad
\frac{\sum_{i,t}m_{it}\ell_{it}}{\sum_iT_i}.
$$

左边先逐回答平均，再跨回答平均；右边在全 batch 有效 token 上平均。即使每个 $$\ell_{it}$$ 一模一样，长短回答在总梯度中的相对权重仍然不同。应当把 reduction 写进算法定义，而不是把它当成不影响结论的代码格式。

### 13.3 终止、截断与 bootstrap

真正到达终止状态时，后续回报为零，可以令 $$V(s_T)=0$$。但“因采样长度预算截断”未必是任务本身结束：若把它当成中途截断，就需要明确 bootstrap 的价值；若训练目标把最大长度定义为硬终止，就应按这个任务定义处理。两种建模都可以讨论，但不能在计算 return、GAE、奖励和 mask 时混用。

### 13.4 至少检查这些量，而不是只看总 reward

- 策略比率的分布与裁剪比例，观察旧数据是否已经偏离当前策略太远。
- 对旧策略的 KL 与对参考策略的 KL，分别观察局部更新幅度和长期行为漂移。
- 价值误差、advantage 的尺度，以及它们随训练的变化。
- 每组奖励方差、全对/全错组比例，确认组相对训练究竟有多少有效信号。
- 回答长度、截断比例及不同长度样本的梯度权重，避免把变长或长度偏置误读为能力改善。
- 多个独立评估维度，检查奖励提高是否真的对应期望的行为改善。

这些指标不自动给出最优超参数，但能帮助区分目标设计、估计方差、策略漂移和实现错误，而不是把所有训练异常都归因于“强化学习不稳定”。


## 14. 把算法放回同一张数学坐标系

如果只记算法名字，很容易把不同层次的修改混在一起。读一个新方法时，我更倾向于依次检查以下几件事。

第一，**随机变量与目标是什么**。是完整回答的期望评分，还是平均 token 的目标？是固定数据的偏好似然，还是当前策略下的在线期望？目标写成相似的求和形式，并不表示采样测度相同。

第二，**梯度乘上的评价信号从哪里来**。Monte Carlo return、价值网络估计的 advantage、同题多个回答的组相对分数各有代价。GRPO 省掉 critic，并没有消灭统计估计问题，而是把一部分问题移到了组大小、奖励分布和组间权重上。

第三，**重要性比率修正了哪一层分布**。一个 token 的条件比率、整个回答的乘积比率、对数比率的长度平均，作用对象和统计性质不同。把某个式子命名为 importance ratio，不足以证明它是某个原目标的无偏估计。

第四，**哪些量被停止梯度，哪些量正在变化**。旧策略、参考策略、价值目标、组均值、组标准差、截断权重，每个量的梯度路径都会影响最终算法。前向数值相近的两个 loss，反向传播可以完全不同。

第五，**裁剪、归一化和过滤改变了什么**。裁剪常用偏差换取稳定性；归一化既影响数值尺度，也可能重新分配问题和回答的权重；过滤零方差组提高有效样本比例，同时改变被训练数据的分布。这些选择应当从优化目标、梯度估计和数据分布三个角度一起解释。

### 14.1 对生成后训练与 Agent 的启发

这些推导不依赖动作必须是文字，但迁移到视频生成或 Agent 时，必须重新确认轨迹概率的定义。

对于 Agent，轨迹中还有工具返回与环境状态。若环境转移不显式依赖策略参数，轨迹求导时这些转移项仍然消失；然而环境反馈和长程奖励会改变 return 的方差与信用分配难度。一个最终成功标签不能自动说明中间哪一步工具调用起了作用，价值估计与过程反馈的价值也因此更加具体。

对于扩散生成，随机去噪转移可以构成轨迹，分数函数方法可用于相应的条件密度。但确定性采样步骤不能不加说明地当成普通随机策略密度；也不能直接把文本 token 的概率比率挪过来。需要先写清采样随机性、转移密度、参考路径分布，再判断在哪些步骤估计、重加权或裁剪梯度。数学框架可以复用，随机变量的定义不能省略。

从这个角度看，PPO 与 GRPO 最值得掌握的不是某一行固定公式，而是公式背后的设计顺序：先定义目标和采样过程，再选择估计器，最后说明为了稳定性与成本引入了哪些近似。把每一次近似放在明处，才有可能判断一个新算法究竟解决了哪一个问题。


## 15. 参考文献

以下列出本文涉及算法的一手资料；日期标注首次公开年份，涉及后续修订时以链接页面为准。本文选择这些方法是为了展示数学设计上的差异，不是按发布时间罗列全部工作，也不意味着某一种方法在所有任务上最优。

1. Williams, 1992. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696). REINFORCE。
2. Schulman et al., 2015. [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477). TRPO。
3. Schulman et al., 2015. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). GAE。
4. Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). PPO。
5. Ouyang et al., 2022. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). 语言模型 RLHF 的代表性实践。
6. Rafailov et al., 2023. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). DPO。
7. Shao et al., 2024. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). GRPO。
8. Ahmadian et al., 2024. [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740). RLOO。
9. Yu et al., 2025. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476). DAPO。
10. Liu et al., 2025. [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783). Dr. GRPO。
11. Zheng et al., 2025. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071). GSPO。
