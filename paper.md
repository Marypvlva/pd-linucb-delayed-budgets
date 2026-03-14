% Skeleton LaTeX for the paper (MOTOR-style)
% Save as: main.tex
\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian,english]{babel}
\usepackage{siunitx}
\sisetup{detect-all}
\usepackage{subcaption}
\usepackage{float}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{enumitem}

\theoremstyle{plain}
\newtheorem{proposition}{Утверждение}
\renewcommand{\proofname}{Доказательство}

\DeclareMathOperator{\clip}{clip}

\geometry{margin=2.2cm}

\newcommand{\inputifexists}[1]{%
  \IfFileExists{#1}{\input{#1}}{\textbf{[Missing file: #1 (generate paper artifacts)]}}%
}

\title{Contextual bandits with a budget constraint and delayed feedback:\\
primal--dual LinUCB and large-scale semi-synthetic experiments on Criteo Attribution}
\author{Мария Павлова}
\date{}

\begin{document}
\maketitle

\begin{abstract}
We study a stochastic online optimization problem in which an agent sequentially selects an action based on an observed context, receives a random reward, and incurs a cost under a global budget constraint. Unlike in classical contextual bandits, reward feedback may be delayed, which complicates both learning and budget control. We investigate a primal--dual variant of Disjoint LinUCB in which the budget constraint is handled through an adaptive dual variable, interpreted as the resource's shadow price. Budget feasibility is enforced mechanically via a stop-at-budget feasible-set protocol (no budget violations).

For reproducible large-scale evaluation, we build a \emph{semi-synthetic} simulator based on the full Criteo Attribution dataset with memmap-based storage. Contexts are sampled from the logged feature distribution. Rewards are generated from an arm-specific linear conditional mean model $\mu_a(x)=\clip(\theta_a^\top x,0,1)$ with $r\sim\mathrm{Bernoulli}(\mu_a(x))$, where $\theta_a$ are fit offline by ridge regression. Costs used by the online budget controller are \emph{arm-level mean costs} computed from the log's \texttt{cost} field and normalized to mean $\approx 1$. We compare the adaptive primal--dual mechanism against cost-aware heuristics of the form $\mathrm{UCB}-\gamma c$ and show empirically that a fixed penalty $\gamma$ is sensitive to the budget regime, while PD-LinUCB is more robust across regimes.
\end{abstract}

\section{Introduction}

Contextual multi-armed bandits and online stochastic optimization arise in sequential decision-making problems where actions must be chosen based on the observed context $x_t$ while balancing exploration and exploitation. Typical applications include ad allocation and bidding, recommendation systems, experiment management, and computing resource allocation, where at each step an action is selected, a random reward is generated, and a limited resource is consumed \cite{harsha2025practical_contextual_discrete_constrained,simchi2025blind_network_bwk_switches}.

In practice, two factors make this problem substantially more challenging. First, actions incur costs, so the optimization objective must account for a resource constraint, as in Bandits with Knapsacks (BwK) and contextual BwK (CBwK) \cite{badanidiyuru2013bwk,agrawal2016cbwk}. Second, rewards are often observed \emph{with delay}, leading to delayed-feedback bandit problems \cite{vernade2017conversions,hoeven2023unified}. Together, these effects create a mismatch between immediately observed costs and delayed reward information, which may render classical algorithms such as LinUCB unstable and/or require manual tuning of cost penalties.

We consider a sequential decision process over rounds $t=1,\dots,T$. At round $t$, the learner observes a context $x_t\in\mathbb{R}^d$ and selects an action $a_t\in\{1,\dots,K\}$. This choice yields a reward $r_t\in[0,1]$ and incurs a cost $c(a_t)\ge 0$. The objective is to maximize the total reward subject to a total budget constraint
\[
\sum_{t=1}^{\tau} c(a_t)\le B,
\]
where $\tau$ denotes the stopping time induced by the stop-at-budget protocol. Reward feedback is delayed: the reward $r_t$ becomes available to the algorithm only at time $t+D_t$, where $D_t\in\{0,1,2,\dots\}$.

In this work, we take Disjoint LinUCB as the base contextual method and introduce an adaptive dual variable $\lambda_t\ge 0$, interpreted as the shadow price of the resource. Actions are selected according to the Lagrangian score
\[
a_t=\arg\max_{a\in\mathcal{A}_t}\big(\mathrm{UCB}_t(x_t,a)-\lambda_t\,c(a)\big).
\]
A key component of the method is a dynamic feasible spending target,
\[
\mathrm{spent}_{t-1}=\sum_{s=1}^{t-1}c(a_s),\qquad
b_t=\frac{B-\mathrm{spent}_{t-1}}{T-t+1},\qquad
\lambda_{t+1}=\big[\lambda_t+\eta\,(c(a_t)-b_t)\big]_+.
\]
Under delayed feedback, we use a \emph{design-now, reward-later} update scheme: the design matrix $A_{a_t}$ is updated immediately after action selection, whereas the response vector $b_{a_t}$ is updated only when the delayed reward $r_t$ becomes available after $D_t$ steps. This is a practical engineering heuristic; it does not inherit standard LinUCB confidence semantics under delay.

\paragraph{Contributions.}
\begin{itemize}
\item A contextual bandit formulation with a budget constraint and delayed feedback under a strict stop-at-budget feasible-set protocol.
\item A practical PD-LinUCB implementation with delayed rewards and a dynamic spending target.
\item A semi-synthetic Criteo-based benchmark with reproducible memmap artifacts.
\item A systematic empirical comparison against LinUCB, fixed-penalty cost-aware heuristics, and a context-free ablation.
\end{itemize}

\section{Problem Statement}
\label{sec:problem}

We consider a horizon $T\in\mathbb{N}$, a set of actions of size $K$, and a total budget $B>0$. At each round $t=1,\dots,T$, the algorithm observes a context $x_t\in\mathbb{R}^d$ and selects an action $a_t\in\mathcal{A}=\{1,\dots,K\}$. Selecting an action yields a stochastic reward $r_t\in[0,1]$ and incurs a cost $c(a_t)\ge 0$. In this work, the cost depends only on the action and is assumed to be known at selection time.

The reward $r_t$ is observed with delay $D_t\in\{0,1,2,\dots\}$. In the \texttt{no-delay} ablation, we set $D_t\equiv 0$.

Let $\mathrm{spent}_t=\sum_{s=1}^{t}c(a_s)$ denote the cumulative cost. We use the \emph{stop-at-budget} protocol, which prohibits executing an action if it would exceed the budget. At round $t$, the feasible set is
\[
\mathcal{A}_t=\{a\in\mathcal{A}: \mathrm{spent}_{t-1}+c(a)\le B\}.
\]
If $\mathcal{A}_t=\varnothing$, the process stops.

\begin{proposition}[Budget adherence]
Under the stop-at-budget protocol, the cumulative cost never exceeds the budget.
\end{proposition}

\begin{proof}
Each executed action satisfies $\mathrm{spent}_{t-1}+c(a_t)\le B$ by construction. Therefore the cumulative cost remains feasible at every executed step.
\end{proof}

We use an arm-specific linear conditional mean model
\[
\mu_a(x)=\mathbb{E}[r\mid x,a]=\clip(\theta_a^\top x,\,0,\,1),
\]
and generate rewards as
\[
r_t \sim \mathrm{Bernoulli}\big(\mu_{a_t}(x_t)\big).
\]

\section{Algorithms}
\label{sec:algorithms}

All methods are evaluated under the same stop-at-budget feasible-set protocol.

\subsection{Disjoint LinUCB}

Disjoint LinUCB maintains a separate ridge model for each action. Under delayed feedback, we update the design matrix immediately after action selection and the response vector only when the reward arrives.

\paragraph{Remark on censored zeros.}
In the conversion-style simulator, non-conversions are modeled as censored zeros that become observable only after an observation window $W$. Under ridge sufficient statistics, a zero label contributes $x\cdot 0$ to the response vector $b$, so it changes the estimate only through the design matrix. Since our implementation updates the design matrix at action time, confirmation of a delayed zero typically produces no additional state change at arrival time. Consequently, in this prototype the learning dynamics are mainly driven by delayed positive rewards.

\subsection{Primal--Dual LinUCB}

PD-LinUCB augments LinUCB with a dual variable $\lambda_t$ and uses the score
\[
a_t=\arg\max_{a\in\mathcal{A}_t}\Big(\mathrm{UCB}_t(x_t,a)-\lambda_t\,c(a)\Big).
\]

\subsection{Cost-aware baselines}

We compare against:

\[
\text{CostNormUCB[ratio]}:\quad
a_t=\arg\max_{a\in\mathcal{A}_t}\frac{\mathrm{UCB}_t(x_t,a)}{c(a)+\varepsilon},
\]

\[
\text{CostNormUCB[sub]}:\quad
a_t=\arg\max_{a\in\mathcal{A}_t}\big(\mathrm{UCB}_t(x_t,a)-\gamma c(a)\big).
\]

\subsection{Context-free PD-BwK}

We also include a context-independent primal--dual baseline that ignores $x_t$ and learns only arm-level empirical mean rewards.

\section{Experiments}
\label{sec:experiments}

\subsection{Real Costs and Delays}

We fix discretization $\Delta=3600$ seconds and an observation window $W$ such that
\[
\frac{W}{\Delta}=5000.
\]
Thus $W=18{,}000{,}000$ seconds ($\approx 208.3$ days). For non-converted events, the simulator returns a censored delay $D=\lceil W/\Delta\rceil$.

Costs used by the online controller are arm-level means derived from the logged \texttt{cost} field and normalized so that the average cost is approximately 1.

\subsection{Main comparison}

\begin{table}[t]
\centering
\caption{Results for $T=5000$ and $\rho=0.7$ under stop-at-budget. Mean values over $N=10$ seeds with $95\%$ confidence intervals.}
\label{tab:main_ci_T5000}
\inputifexists{paper_artifacts/tables/main_ci.tex}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.75\linewidth]{paper_artifacts/figures/baselines_cum_reward_full4_arm.png}
\caption{Cumulative reward (example run).}
\label{fig:cum_reward}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.75\linewidth]{paper_artifacts/figures/baselines_cum_cost_full4_arm.png}
\caption{Cumulative cost (example run).}
\label{fig:cum_cost}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.80\linewidth]{paper_artifacts/figures/baselines_cum_reward_mean_ci.png}
\caption{Cumulative reward (mean $\pm$ 95\% CI over seeds).}
\label{fig:cum_reward_mean_ci}
\end{figure}

\subsection{Gamma sweep}

\begin{figure}[t]
\centering
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth,height=0.28\textheight,keepaspectratio]{paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_reward.png}
  \caption{Total reward (mean $\pm$ 95\% CI).}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth,height=0.28\textheight,keepaspectratio]{paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_spent.png}
  \caption{Budget usage: $\mathrm{spent}/B$ (mean $\pm$ 95\% CI).}
\end{subfigure}
\caption{Sweep over the penalty $\gamma$ for CostNormUCB[sub] with $\rho=0.7$ and $T=5000$. The x-axis uses a symlog scale to include $\gamma=0$.}
\label{fig:gamma_sweep}
\end{figure}

\subsection{Delay vs no-delay}

\begin{table}[t]
\centering
\caption{Delay ablation: empirical delays vs no-delay ($D_t\equiv 0$).}
\label{tab:delay_ablation}
\inputifexists{paper_artifacts/tables/delay_ablation.tex}
\end{table}

\subsection{Budget sweep}

\begin{table}[t]
\centering
\caption{Budget sweep over $\rho=B/T$: PD-LinUCB vs best fixed-penalty baseline with $\gamma^\star(\rho)$.}
\label{tab:budget_sweep}
\inputifexists{paper_artifacts/tables/budget_sweep.tex}
\end{table}

\subsection{Reproducibility settings}

\begin{table}[H]
\centering
\caption{Default hyperparameters and settings.}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Number of arms $K$ & $50$ \\
Context dimension $d$ & $65$ \\
Horizon $T$ & $5000$ \\
Budget mode $\rho=B/T$ & $\{0.40, 0.55, 0.70, 0.85\}$ \\
Regularization $\lambda$ & $1.0$ \\
LinUCB $\alpha_{\text{lin}}$ & $1.0$ \\
PD-LinUCB $\alpha_{\text{pd}}$ & $1.5$ \\
PD-LinUCB step $\eta$ & $0.05$ \\
Sampling $\Delta$ & $3600$ seconds \\
Observation window $W$ & $18{,}000{,}000$ sec ($W/\Delta=5000$) \\
$D_{\max}$ & $5000$ steps \\
\bottomrule
\end{tabular}
\end{table}

\appendix
\section{Pseudocodes}

\paragraph{CostNormUCB[ratio].}
At each round, form the feasible set $\mathcal{A}_t$, stop if it is empty, and otherwise select
\[
a_t \leftarrow \arg\max_{a\in\mathcal{A}_t} \frac{\mathrm{UCB}_t(x_t,a)}{c(a)+\varepsilon}.
\]

\paragraph{CostNormUCB[sub].}
At each round, form the feasible set $\mathcal{A}_t$, stop if it is empty, and otherwise select
\[
a_t \leftarrow \arg\max_{a\in\mathcal{A}_t} \big(\mathrm{UCB}_t(x_t,a)-\gamma c(a)\big).
\]

\bibliographystyle{plain}
\bibliography{refs}

\end{document}
