% Skeleton LaTeX for the paper (MOTOR-style)
% Save as: main.tex
\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian,english]{babel}
% --- tables with SI-style numbers (optional but more convenient) ---
\usepackage{siunitx}
\sisetup{detect-all}
\usepackage{subcaption} % for subfigure
\usepackage{float}
\usepackage{amsmath,amssymb,amsthm}

\theoremstyle{plain}
\newtheorem{proposition}{Утверждение} % or use [section] to number by section
\renewcommand{\proofname}{Доказательство} % avoid the default "Proof." label
\DeclareMathOperator{\clip}{clip}
\usepackage{mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{enumitem}

\geometry{margin=2.2cm}
\newcommand{\inputifexists}[1]{\IfFileExists{#1}{\input{#1}}{\textbf{[Missing artifact: #1]}}}

\title{Contextual bandits with a budget constraint and delayed feedback:\\
primal--dual LinUCB and a large-scale semi-synthetic benchmark on Criteo Attribution}
\author{Мария Павлова}
\date{}

\begin{document}
\maketitle

\begin{abstract}
We study a stochastic online optimization problem in which an agent sequentially selects an action based on an observed context, receives a random reward, and incurs a cost under a global budget constraint. Unlike in classical contextual bandits, reward feedback may be delayed, which complicates both learning and budget control. We propose and investigate a primal--dual variant of Disjoint LinUCB in which the budget constraint is handled through an adaptive dual variable, interpreted as the resource's shadow price, while model updates are performed as delayed feedback arrives. To ensure reproducibility, we build a scalable semi-synthetic benchmark on the full Criteo Attribution dataset, using memmap-based storage together with a stop-at-budget protocol that guarantees zero budget violations.

In experiments on an arm-specific contextual environment, we use a linear conditional mean model
$\mu_a(x)=\clip(\theta_a^\top x,0,1)$ and generate rewards according to
$r\sim\mathrm{Bernoulli}(\mu_a(x))$,
with arm-specific parameters $\theta_a$ estimated by ridge regression on the temporal training split. We also compare our method against a fixed Lagrangian-penalty baseline of the form $\mathrm{UCB}-\gamma c$ and show that its performance is highly sensitive to both the choice of $\gamma$ and the budget regime. In particular, the optimal $\gamma^\star$ changes with the ratio $B/T$, which motivates nested tuning on a separate split/seed set before held-out evaluation. Overall, the results show that the adaptive primal--dual mechanism provides a robust alternative to simple cost-aware heuristics.
\end{abstract}


\section{Introduction}

Contextual multi-armed bandits and online stochastic optimization arise in sequential decision-making problems where actions must be chosen based on the observed context $x_t$ while balancing exploration and exploitation. Typical applications include ad allocation and bidding, recommendation systems, experiment management, and computing resource allocation, where at each step an action is selected, a random reward is generated, and a limited resource (such as budget, time, impressions, or tokens) is consumed \cite{harsha2025practical_contextual_discrete_constrained,simchi2025blind_network_bwk_switches}.

In practice, two factors make this problem substantially more challenging. First, actions incur costs, so the optimization objective must account for a resource constraint, as in Bandits with Knapsacks (BwK) and contextual BwK (CBwK) \cite{badanidiyuru2013bwk,agrawal2016cbwk}. Second, rewards are often observed \emph{with delay} (for example, conversions may occur hours or days after the action), leading to delayed-feedback bandit problems \cite{vernade2017conversions,hoeven2023unified}. Together, these effects create a mismatch between immediately observed costs and delayed reward information, which may render classical algorithms such as LinUCB unstable and/or require manual tuning of cost penalties.

We consider a sequential decision process over rounds $t=1,\dots,T$. At round $t$, the learner observes a context $x_t\in\mathbb{R}^d$ and selects an action $a_t\in\{1,\dots,K\}$. This choice yields a reward $r_t\in[0,1]$ and incurs a cost $c(a_t)\ge 0$. The objective is to maximize the total reward subject to a total budget constraint
\[
\sum_{t=1}^{\tau} c(a_t)\le B,
\]
where $\tau$ denotes the stopping time induced by the stop-at-budget protocol, that is, an action is not executed if it would cause the cumulative cost to exceed the budget. Reward feedback is delayed: the reward $r_t$ becomes available to the algorithm only at time $t+D_t$, where $D_t\in\{0,1,2,\dots\}$.

Classical BwK and CBwK methods provide resource control and regret guarantees, but they are typically analyzed in delay-free settings or under more aggregated feedback models \cite{badanidiyuru2013bwk,agrawal2016cbwk}. By contrast, the literature on delayed bandits studies how delays affect regret and how UCB- or FTRL-type methods should be modified, but usually without an explicit budget constraint \cite{vernade2017conversions,hoeven2023unified}. There is also a substantial literature on primal--dual methods for online constrained optimization, in which the dual variable plays the role of the resource's shadow price and is adapted online \cite{mahdavi2012trading,li2021primaldual,liu2025augmented_lagrangian_tv_oco}. However, in many practical settings, especially in advertising, the following ingredients appear simultaneously: (i) a global budget constraint, (ii) delayed reward feedback, (iii) the need for scalable and reproducible empirical evaluation on real-world data, and (iv) simple heuristics of the form $\mathrm{UCB}-\gamma c$, whose performance is highly sensitive to the choice of $\gamma$ and to the budget regime. This combination calls for an explicit separation between the cost-control mechanism, which must operate online based on immediately observed costs, and the reward-learning mechanism, which must rely on delayed observations, together with an evaluation protocol that guarantees zero budget violations.

In this work, we take Disjoint LinUCB as the base contextual method and introduce an adaptive dual variable $\lambda_t\ge 0$, interpreted as the shadow price of the resource. Actions are selected according to the Lagrangian score
\[
a_t=\arg\max_{a}\big(\mathrm{UCB}_t(x_t,a)-\lambda_t\,c(a)\big).
\]
A key component of the method is a dynamic feasible spending target, defined as the remaining budget divided by the remaining horizon:
\[
\mathrm{spent}_{t-1}=\sum_{s=1}^{t-1}c(a_s),\qquad
b_t=\frac{B-\mathrm{spent}_{t-1}}{T-t+1},\qquad
\lambda_{t+1}=\big[\lambda_t+\eta\,(c(a_t)-b_t)\big]_+.
\]
Under delayed feedback, we use a \emph{design-now, reward-later} update scheme: the design matrix $A_{a_t}$ is updated immediately after action selection, whereas the response vector $b_{a_t}$ is updated only when the delayed reward $r_t$ becomes available after $D_t$ steps. Budget feasibility is enforced by the stop-at-budget protocol, which halts the process before any action that would violate the budget constraint.
\paragraph{Contributions.}
\begin{itemize}
  \item \textbf{Unified problem formulation:} a contextual bandit with a budget constraint and delayed feedback
  under a \emph{stop-at-budget} protocol that guarantees zero budget violations during execution.

  \item \textbf{PD-LinUCB algorithm for delayed feedback:} a primal--dual variant of Disjoint LinUCB with a dynamic spending target
  $b_t=(B-\mathrm{spent}_{t-1})/(T-t+1)$ and delayed updates implemented via the \emph{design-now, reward-later} scheme.

  \item \textbf{Semi-synthetic benchmark based on Criteo:} a reproducible online environment built on the full
  Criteo Attribution dataset (16.47M events) stored in \texttt{memmap} format, with a temporal train/test split, arm-level costs from the \texttt{cost} field,
  and delays computed from \texttt{conversion\_timestamp}-\texttt{timestamp} with discretization and observation-window censoring.

  \item \textbf{Systematic empirical evaluation:} comparison of PD-LinUCB with LinUCB, cost-aware heuristics, and a context-independent PD-BwK baseline,
  including held-out simulator diagnostics, a sensitivity analysis of the fixed penalty $\gamma$ in $\mathrm{UCB}-\gamma c$ across different budget regimes,
  and reporting $95\%$ confidence intervals over multiple seeds (reward, spent$/B$, stopping time).
\end{itemize}


\section{Problem Statement}
\label{sec:problem}

We consider a horizon $T\in\mathbb{N}$, a set of actions (arms) of size $K$, and a total budget $B>0$.
At each round $t=1,\dots,T$, the algorithm observes a context $x_t\in\mathbb{R}^d$ and selects an action
$a_t\in\mathcal{A}=\{1,\dots,K\}$.
Selecting an action yields a stochastic reward $r_t\in[0,1]$ and incurs a cost $c(a_t)\ge 0$.
In this work, the cost depends only on the action (arm-dependent cost) and is assumed to be known at the time of selection.

The reward $r_t$ is observed with delay $D_t\in\{0,1,2,\dots\}$: the value of $r_t$ becomes available to the algorithm
only at time $t+D_t$, and only then can it be used to update the model.
In the \texttt{no-delay} ablation, we set $D_t\equiv 0$.
In the raw dataset, the value $D_t=-1$ may appear, meaning ``no observation / outside the window'';
when constructing the online environment, this value is \emph{not} interpreted as zero delay.

Let $\mathrm{spent}_t=\sum_{s=1}^{t}c(a_s)$ denote the cumulative cost (with $\mathrm{spent}_0=0$).
We use the \emph{stop-at-budget} protocol, which prohibits executing an action if it would exceed the budget.
At round $t$, action $a_t$ is executed only if
\[
\mathrm{spent}_{t-1}+c(a_t)\le B.
\]
Let $\tau$ denote the number of actions executed before stopping. Then
\[
\tau=\max\Big\{t:\mathrm{spent}_t\le B\Big\}.
\]

The objective is to maximize the expected cumulative reward until stopping:
\[
R(\pi)=\mathbb{E}\Big[\sum_{t=1}^{\tau(\pi)} r_t\Big],\qquad
\tau(\pi)=\max\Big\{\tau:\sum_{t=1}^{\tau} c(a_t)\le B\Big\}.
\]
The optimal feasible policy is
$\pi^\star\in\arg\max_{\pi\in\Pi} R(\pi)$, and the regret of an algorithm $\mathcal{A}$ is defined as
\[
\mathrm{Reg}(T,B)= R(\pi^\star) - R(\mathcal{A}).
\]

When $D_t\equiv 0$, the problem reduces to contextual bandits with a budget constraint (CBwK).
When $B\to\infty$, the problem reduces to contextual bandits with delayed feedback.
When contexts are absent ($x_t\equiv x$), the setting reduces to BwK with delays.

\begin{proposition}[Budget adherence in stop-at-budget mode]
\label{prop:budget}
Under the stop-at-budget protocol, the cumulative cost never exceeds the budget:
$\sum_{t=1}^{\tau} c(a_t)\le B$ for any run of the algorithm and for any delay realization.
\end{proposition}

\begin{proof}
By definition of the stop-at-budget protocol, action $a_t$ is executed only if the current cumulative cost
$\mathrm{spent}_{t-1}=\sum_{s< t}c(a_s)$ satisfies $\mathrm{spent}_{t-1}+c(a_t)\le B$.
Therefore, by induction, $\mathrm{spent}_t\le B$ for all $t\le \tau$, and the process stops exactly when the next action would violate the budget constraint.
\end{proof}

We use an arm-specific linear conditional mean model
\begin{equation}
\mu_a(x)=\mathbb{E}[r\mid x,a]=\mathrm{clip}(\theta_a^\top x,\,0,\,1),
\label{eq:mu_linear_clip}
\end{equation}
where $\mathrm{clip}(u,0,1)=\min\{1,\max\{0,u\}\}$.
Rewards are generated as
\begin{equation}
r_t \sim \mathrm{Bernoulli}\big(\mu_{a_t}(x_t)\big).
\label{eq:bern_linear}
\end{equation}

The parameters $\theta_a$ are calibrated on the logged training split via ridge regression:
\begin{equation}
\widehat{\theta}_a=\arg\min_{\theta}\sum_{i:A_i=a}(R_i-\theta^\top X_i)^2+\lambda\|\theta\|_2^2.
\label{eq:ridge}
\end{equation}

The main benchmark uses this clipped-linear simulator. In addition, we fit an optional arm-wise logistic simulator on the same training split and report held-out calibration diagnostics for it.



\section{Related Work}
\label{sec:related}

Our setting lies at the intersection of three research directions: (i) resource-constrained bandits (BwK and their contextual variants), (ii) primal--dual and Lagrangian methods for online optimization with long-term constraints, and (iii) bandits with delayed feedback, particularly those motivated by delayed conversions.

\subsection{BwK, CBwK, and Linear Constraints}

The \emph{Bandits with Knapsacks} (BwK) framework formalizes learning under limited resources (e.g., a budget constraint) and has become the standard model for constrained bandit problems \cite{badanidiyuru2013bwk,badanidiyuru2018bwk,elumar2025anytime}. Early extensions consider more general objective functions and constraint structures, including settings with concave rewards and convex knapsack constraints \cite{agrawal2014concave}.

Contextual generalizations, known as contextual bandits with knapsacks (CBwK), allow action choices to depend on observed contexts while satisfying a global resource constraint \cite{agrawal2016cbwk,guo2025stochastic_cbwk_smallbudget,zhao2024constrained}. From a practical perspective, computational efficiency is crucial. In particular, the framework developed by Agrawal, Devanur, and Li provides a general reduction-based approach for contextual policies and extends naturally to settings with concave objectives \cite{agrawal2016cbwk}. Another related line of work studies \emph{contextual bandits with linear packing and covering constraints}, where multiple resources are subject to linear constraints. Conceptually, these models are closely related to the stop-at-budget protocol used in our work, which can be viewed as a special case of resource exhaustion under a single constraint \cite{agrawal2014bandits}.

\subsection{Primal--dual and Lagrangian Resource Control}

Primal--dual methods are a standard approach to online optimization with long-term constraints. In these methods, the dual variable acts as the shadow price of a resource and is updated according to the (sub)gradient of the constraint violation, while the primal decision is selected using a Lagrangian score \cite{mahdavi2012trading,miao2025primal_dual_rm_demand_learning}. In the BwK setting, this idea leads to explicit primal--dual algorithms in which the multiplier controls the reward--cost trade-off and enables adaptive budget management during learning \cite{li2021primaldual}. More recent work extends this paradigm to contextual constrained optimization beyond classical CBwK, for example in contextual stochastic combinatorial optimization problems \cite{bouvier2025primal}.

Our approach follows this line of work but focuses on a \emph{simple} and practically reproducible variant for linear contextual models. Specifically, we incorporate a Lagrangian penalty directly into the UCB index and introduce a dynamic spending target based on the remaining budget and remaining horizon. This formulation is particularly convenient for the stop-at-budget protocol and for scalable engineering implementations.

\subsection{Delayed Feedback in Bandits and Delayed Conversions}

Delayed feedback in bandit problems has been studied extensively from both algorithmic and theoretical perspectives. The classical formulation considers online learning or bandits with delayed feedback and analyzes how delays affect regret and algorithm design \cite{joulani2013online,schlisselberg2025improved}. In advertising applications, the delayed conversions model is particularly important: positive rewards (conversions) arrive with delay, while the absence of a conversion corresponds to a censored zero outcome. Such models and the corresponding UCB-based algorithms are analyzed, for example, by Vernade, Cappé, and Perchet \cite{vernade2017conversions}.

Unlike these works, we consider delayed feedback together with an explicit budget constraint. In our setting, the cost of an action is incurred immediately, whereas the reward is observed later. This separation naturally decouples \emph{resource control} from \emph{reward model learning}, which is crucial in practical systems \cite{ji2026inventory_delayed_feedback}.

\subsection{How PD-LinUCB Differs from Prior Work}

Compared with the BwK/CBwK/CBwLC literature \cite{badanidiyuru2013bwk,agrawal2016cbwk,slivkins2023contextual}, our goal is not to establish new regret guarantees for the most general classes of policies or linear constraints. Instead, we focus on a widely used linear contextual model (Disjoint LinUCB) and study how a \emph{dynamic} primal--dual multiplier improves practical performance relative to the static penalty heuristic $\mathrm{UCB}-\gamma c$. In particular, we demonstrate empirically that the dynamic multiplier provides significantly greater robustness across different budget regimes while guaranteeing zero budget violations under the stop-at-budget protocol.

Compared with the literature on delayed bandits \cite{joulani2013online,vernade2017conversions}, our contribution lies in explicitly combining delayed feedback with a budget constraint and implementing the correct update semantics for delayed rewards (the \emph{design-now, reward-later} scheme). This distinction is critical for practical large-scale implementations using real-world logs with realistic costs and delays.






\section{Algorithms}
\label{sec:algorithms}

This section describes the methods compared in our experiments:  
(i) the baseline contextual algorithm Disjoint LinUCB,  
(ii) the proposed PD-LinUCB method with primal--dual (Lagrangian) budget control,  
(iii) simple cost-aware heuristics based on UCB, and  
(iv) a context-independent primal--dual baseline used as an ablation (“no context”).  

All methods assume that the cost $c(a)$ is known at the time of action selection, while the reward $r_t$ may be observed with a delay $D_t$.

\paragraph{Stop-at-budget protocol (common to all algorithms).}
Let $\mathrm{spent}_{t-1}=\sum_{s=1}^{t-1}c(a_s)$ denote the cumulative cost before round $t$.
At round $t$, the feasible set of actions is defined as
\[
\mathcal{A}_t=\{a\in\mathcal{A}: \mathrm{spent}_{t-1}+c(a)\le B\}.
\]
If $\mathcal{A}_t=\varnothing$, the process stops and $\tau=t-1$.
Otherwise, the algorithm selects an action $a_t\in\mathcal{A}_t$ according to its decision rule (maximizing its score or index over the feasible set),
after which the cost is incurred.  
This protocol guarantees $\mathrm{spent}_\tau\le B$ and avoids the “early stopping” artifact that would arise if an infeasible action were selected while feasible alternatives still existed.

\subsection{Disjoint LinUCB (baseline)}
\label{subsec:linucb}

Disjoint LinUCB maintains a separate linear ridge regression model for each action $a\in\{1,\dots,K\}$ and selects actions using an upper confidence bound (UCB) criterion.

\paragraph{Parameter estimation.}
For each action $a$, the algorithm maintains a design matrix and response vector:
\[
A_{a,t}=\lambda I + \sum_{i\in\mathcal{P}_t:\,a_i=a} x_i x_i^\top,
\qquad
b_{a,t}=\sum_{i\in\mathcal{O}_t:\,a_i=a} x_i r_i,
\]
where $\mathcal{P}_t$ is the set of rounds whose actions have already been \emph{committed} by time $t$ (the contexts $x_i$ are known immediately),  
and $\mathcal{O}_t\subseteq\mathcal{P}_t$ is the subset of rounds for which the reward has already been \emph{observed} by time $t$ (accounting for delays).  

The parameter estimate is
\[
\widehat{\theta}_{a,t}=A_{a,t}^{-1}b_{a,t}.
\]

\paragraph{UCB index and action selection.}
Given context $x_t$, the algorithm computes
\begin{equation}
\mathrm{UCB}_t(x_t,a)=
\widehat{\theta}_{a,t}^\top x_t
+\alpha\sqrt{x_t^\top A_{a,t}^{-1}x_t},
\label{eq:ucb}
\end{equation}
and selects
\[
a_t=\arg\max_{a\in\mathcal{A}_t}\mathrm{UCB}_t(x_t,a).
\]

\paragraph{Delayed feedback: \emph{design-now, reward-later}.}

A key implementation detail under delayed feedback is that contexts are available immediately. Therefore, after selecting an action we update the \emph{design matrix} immediately:
\[
A_{a_t}\leftarrow A_{a_t}+x_tx_t^\top.
\]

The \emph{response vector} is updated only when the reward becomes available:
\[
b_{a_t}\leftarrow b_{a_t}+x_t r_t
\quad \text{at time } t+D_t,
\]
including the case $D_t=0$, where the update occurs in the same round.

Conceptually, this corresponds to the fact that exposures (pairs $(a_t,x_t)$) are recorded immediately, while the corresponding labels $r_t$ arrive later.

Note that under delayed feedback the uncertainty radius is computed using the current geometry of contexts before the corresponding rewards are observed. We treat this as a practical engineering heuristic that works well in large-scale empirical settings; a rigorous theoretical treatment of delayed confidence radii is beyond the scope of this work \cite{joulani2013online,hoeven2023unified}.

Under this linear ridge update, a confirmed zero label contributes $x\cdot 0$ to the response vector and therefore does not change the model at arrival time once the design update has already been committed. Accordingly, in the default linear benchmark the delayed learning dynamics are driven mainly by positive outcomes; the optional logistic simulator and logistic policy family are included as a more statistically faithful alternative that updates on both positive and negative labels.
\paragraph{Numerical efficiency.}
For computational efficiency, the inverse matrix $A_{a,t}^{-1}$ is maintained explicitly, and the Sherman--Morrison rank-one update is applied when adding the outer product $x_tx_t^\top$:
\begin{equation}
A_{a}^{-1}\leftarrow A_{a}^{-1}-\frac{A_{a}^{-1}x x^\top A_{a}^{-1}}{1+x^\top A_{a}^{-1}x}.
\label{eq:SM}
\end{equation}

\subsection{Primal--Dual LinUCB (PD-LinUCB)}
\label{subsec:pdlinucb}

The proposed method augments LinUCB with a Lagrangian multiplier $\lambda_t\ge 0$, interpreted as the \emph{shadow price} of the resource, which adapts to the trajectory of budget consumption. Conceptually, the method follows the standard Lagrangian formulation of the form “reward minus price $\times$ cost,” with the dual variable updated online. In the context of BwK/CBwK, this provides a natural practical mechanism for enforcing budget constraints through a primal--dual scheme.

\paragraph{Action selection rule (Lagrangian score).}
At round $t$, the action is selected according to the score
\begin{equation}
a_t=\arg\max_{a\in\mathcal{A}}\Big(\mathrm{UCB}_t(x_t,a)-\lambda_t\,c(a)\Big),
\label{eq:pd_choice}
\end{equation}
where $\mathrm{UCB}_t$ is computed as in~\eqref{eq:ucb}, and $c(a)$ denotes the known cost of action $a$.

\paragraph{Stop-at-budget (feasible set).}
At each round $t$, the feasible set of actions
\[
\mathcal{A}_t=\{a:\mathrm{spent}_{t-1}+c(a)\le B\}
\]
is determined first. If $\mathcal{A}_t=\varnothing$, the process stops. Otherwise, the choice in~\eqref{eq:pd_choice} (and similarly for the baseline methods) is performed by maximizing the score over $a\in\mathcal{A}_t$.

\paragraph{Dual update with a dynamic spending target.}
Instead of using a fixed per-step budget, we define a dynamic spending target that reflects the remaining budget relative to the remaining horizon:
\begin{equation}
b_t=\frac{B-\mathrm{spent}_{t-1}}{T-t+1}.
\label{eq:bt_dynamic}
\end{equation}
The dual variable is then updated via a projected gradient step onto $\mathbb{R}_+$:
\begin{equation}
\lambda_{t+1}=\big[\lambda_t+\eta\,(c(a_t)-b_t)\big]_+.
\label{eq:dual_dynamic}
\end{equation}
Intuitively, if the selected action is more expensive than the target ($c(a_t)>b_t$), the multiplier $\lambda_t$ increases, penalizing costly actions more strongly in subsequent rounds. Conversely, if $c(a_t)<b_t$, the multiplier decreases (down to zero), reducing the penalty.

\paragraph{Delayed feedback: \emph{design-now, reward-later}.}
As in the LinUCB baseline, the context is available immediately. Therefore, once an action is executed we update the design matrix:
\[
A_{a_t}\leftarrow A_{a_t}+x_tx_t^\top,
\]
and update the inverse matrix $A_{a_t}^{-1}$ using~\eqref{eq:SM}.  
The response update
\[
b_{a_t}\leftarrow b_{a_t}+x_tr_t
\]
is deferred until time $t+D_t$ (or applied immediately when $D_t=0$).  
Thus, budget control (through $\lambda_t$ and the stop-at-budget rule) operates fully online, while reward learning relies on delayed feedback.
\subsection{Cost-aware baselines}
\label{subsec:heuristics}

To verify that the performance gain of PD-LinUCB is not merely due to trivial cost awareness, we compare it with two common heuristics that use the same UCB index~\eqref{eq:ucb}.

\paragraph{CostNormUCB[ratio].}
\[
a_t=\arg\max_{a\in\mathcal{A}}
\frac{\mathrm{UCB}_t(x_t,a)}{c(a)+\varepsilon},
\]
where $\varepsilon>0$ prevents division by zero.

\paragraph{CostNormUCB[sub] (fixed penalty).}
\[
a_t=\arg\max_{a\in\mathcal{A}}
\big(\mathrm{UCB}_t(x_t,a)-\gamma\,c(a)\big),
\]
where $\gamma\ge 0$ is a fixed multiplier (a static Lagrangian penalty) that must be tuned.
In the experiments, we perform a sweep over $\gamma$ and report how both total reward and budget utilization vary with this fixed penalty.

Both heuristics use the same \emph{design-now, reward-later} delayed-update scheme for $(A_a,b_a)$ and are evaluated under the same stop-at-budget protocol.

\subsection{Context-independent primal--dual baseline (CF-PD-BwK)}
\label{subsec:cfbwk}

To isolate the contribution of contextual information from that of budget control, we include a baseline that ignores the context $x_t$ and learns only the average reward of each action.
Let $\widehat{\mu}_{a,t}$ denote the empirical mean reward of action $a$ (based on observed rewards, accounting for delays), and let $n_{a,t}$ denote the number of \emph{observed} rewards for action $a$ by time $t$.
The action is then selected using a UCB-style index with a Lagrangian penalty:
\[
a_t=\arg\max_{a\in\mathcal{A}}
\Big(
\widehat{\mu}_{a,t}
+\alpha\sqrt{\tfrac{\log(t+1)}{\max(1,n_{a,t})}}
-\lambda_t\,c(a)
\Big).
\]

The multiplier $\lambda_t$ is updated using the same dynamic rule~\eqref{eq:dual_dynamic} with the spending target $b_t$ from~\eqref{eq:bt_dynamic}.
This baseline helps identify regimes in which performance improvements are driven primarily by differences between arms (and budget management) rather than by contextual adaptation.

\subsection{Summary: Methods Compared}
\label{subsec:methods_summary}

The experiments compare the following methods:
(1) LinUCB, 
(2) PD-LinUCB, 
(3) CostNormUCB[ratio] and CostNormUCB[sub], and 
(4) CF-PD-BwK.

All methods are run on the same sequence of contexts in the same delayed environment and are evaluated using the same metrics (Section~\ref{sec:experiments}) under the stop-at-budget protocol.
\section{Experiments}
\label{sec:experiments}

The goal of these experiments is to empirically compare PD-LinUCB with baseline methods in the contextual bandit problem with a budget constraint and delayed feedback (Section~\ref{sec:problem}) under a strictly enforced feasibility constraint (the stop-at-budget protocol). All methods are evaluated on identical context sequences with identical random seeds. We report mean values and $95\%$ confidence intervals across independent runs.

\subsection{Data and Feature Generation: Criteo Attribution}
\label{subsec:data}

We use the \emph{Criteo Attribution} dataset (size $n=16\,468\,027$ rows, archive \texttt{criteo\_attribution\_dataset.tsv.gz}) as a source of realistic distributions of contexts, costs, and delays. Each row corresponds to an event (contact/impression/click) and contains a campaign identifier, a binary conversion label, timestamps, and contextual features. To reduce leakage between simulator fitting and evaluation, we construct a \emph{temporal} train/test split: the simulator parameters are fit on the earlier $80\%$ of rows, while online evaluation samples contexts from the later $20\%$.

The context is constructed as a vector $x\in\mathbb{R}^d$ of dimension $d=65$:
$64$ hashed binary features derived from categorical fields (feature hashing) and one numerical feature
$\log(1+\max\{0,\texttt{time\_since\_last\_click}\})$.
The preprocessed arrays are stored in memmap format:
\[
X\in\mathbb{R}^{n\times d}\ (\texttt{X.npy}),\quad
A\in\{0,\dots,K-1\}^n\ (\texttt{A.npy}),\quad
R\in\{0,1\}^n\ (\texttt{R.npy}),\quad
C\in\mathbb{R}_+^{n}\ (\texttt{C.npy}),\quad
D\in\mathbb{Z}^{n}\ (\texttt{D.npy}).
\]

Actions correspond to advertising campaigns. For a fixed number of arms, we map campaigns to arms via
\[
a = \mathrm{hash}(\mathrm{campaign}) \bmod K,\qquad K=50.
\]

\noindent\textbf{Replicated data artifact.}
All main experiments use the memmap artifact
\texttt{data/processed/criteo\_full\_k50\_d64\_real\_split80/}
(including \texttt{split.npy}, real arm-dependent costs \texttt{costs\_by\_arm.npy}, a global delay pool \texttt{delays\_pos.npy}, and arm-conditional positive-delay pools \texttt{delays\_pos\_by\_arm.npz}).

\subsection{Real Costs and Delays: Observation Window Discretization and Censoring}
\label{subsec:real_cost_delay}

The dataset contains the fields \texttt{cost}, \texttt{timestamp}, and \texttt{conversion\_timestamp}. We use the \texttt{cost} field as the resource consumption and compute delays from the difference between event and conversion timestamps.

We fix a discretization step $\Delta>0$ (in the experiments $\Delta=3600$ seconds, i.e., one hour) and an observation window $W>0$.
In the published memmap artifact, the parameters are chosen such that
\[
\frac{W}{\Delta}=5000,
\]
meaning that the \emph{censoring delay} for the absence of a conversion is $D=5000$ steps.
With $\Delta=3600$, this corresponds to $W=18{,}000{,}000$ seconds ($\approx 208.3$ days).

For an event with a conversion, the delay in steps is defined as
\[
D_i=\min\Big\{D_{\max},\ \Big\lceil\frac{\texttt{conversion\_timestamp}_i-\texttt{timestamp}_i}{\Delta}\Big\rceil\Big\},
\]
where $D_{\max}$ denotes the maximum delay in steps (in our experiments, $D_{\max}=5000$).

For non-converted events, we apply observation-window censoring:
\[
D_i = \min\{D_{\max}, \lceil W/\Delta\rceil\}.
\]
Thus, a negative outcome is confirmed only after the observation window $W$ expires (which corresponds to $D=5000$ steps). In the online environment, negative outcomes are therefore not instantaneous: the algorithm receives confirmation $r=0$ only after a delay. Positive delays are sampled either from the global empirical pool or, in the improved simulator, from arm-conditional empirical pools.

\paragraph{Note on special delay values.}
In the raw data, the value $D=-1$ may occur, meaning ``outside the observation window'' or ``no observation''. In our online environment this value is \emph{not} interpreted as zero delay. The \texttt{no-delay} ablation is defined separately by setting $D_i\equiv 0$ for all events.
\paragraph{Cost.}
The cost $c$ is derived from the \texttt{cost} field. To ensure comparability across budget regimes, we use an arm-dependent cost: for each arm $a$, the average cost is computed from the logs and then used as $c(a)$ in the simulator. We further normalize costs by the average cost across arms so that $\mathbb{E}[c(a)]\approx 1$, ensuring that the budget parameter $\rho$ has a consistent interpretation across experiments.

\subsection{Online Environment and Reward Generation}
\label{subsec:env}

The online environment is implemented as a simulator. At each round $t$, a context $x_t$ (a randomly sampled row from $X$) is provided to the algorithm. The algorithm selects an action $a_t$, after which the environment returns the triple $(r_t,c(a_t),D_t)$.

The delay $D_t$ is sampled from the empirical distribution of positive conversion delays (for $r=1$). For $r=0$, we use a censored delay $D_t=\lceil W/\Delta\rceil$.

The conversion probability is generated according to an arm-specific linear conditional mean model
\[
\mu_a(x)=\clip(\theta_a^\top x,0,1),\qquad r\sim\mathrm{Bernoulli}(\mu_a(x)),
\]
where the parameters $\theta_a$ are calibrated using ridge regression on the training split for each arm (Section~\ref{sec:problem}). This design yields a controlled semi-synthetic environment that preserves realistic distributions of contexts, costs, and delays while ensuring reproducible reward generation. We additionally fit an arm-wise logistic simulator on the same training split and evaluate its held-out calibration, but the main policy comparisons in this paper use the default clipped-linear simulator unless noted otherwise.

\subsection{Held-Out Simulator Diagnostics}
\label{subsec:diagnostics}

To assess how well the fitted simulator matches held-out rows, we report calibration and split-stability diagnostics on the test split. The diagnostics include Brier score and log loss for the reward model, the train-vs-test CDF of positive delays, and arm-level train-vs-test comparisons for average delay and normalized arm cost.

\begin{table}[t]
\centering
\caption{Held-out simulator diagnostics on the temporal test split. Lower Brier/log loss are better.}
\label{tab:simulator_diagnostics}
\inputifexists{paper_artifacts/tables/simulator_diagnostics.tex}
\end{table}

\begin{figure}[t]
\centering
\IfFileExists{paper_artifacts/figures/simulator_calibration.png}
  {\includegraphics[width=0.72\linewidth]{paper_artifacts/figures/simulator_calibration.png}}
  {\fbox{\parbox{0.72\linewidth}{Missing artifact: paper\_artifacts/figures/simulator\_calibration.png}}}
\caption{Held-out calibration of the simulator on test rows. Quantile bins are used to avoid sparse-bin artifacts in the rare-event tail.}
\label{fig:simulator_calibration}
\end{figure}

\subsection{Experiment Protocol and Metrics}
\label{subsec:protocol}

All experiments use the stop-at-budget protocol: an action at round $t$ is executed only if
\[
\mathrm{spent}_{t-1}+c(a_t)\le B,
\]
otherwise the process terminates. For a given horizon $T$, we parameterize the budget regime as
\[
\rho=\frac{B}{T}\qquad\Longleftrightarrow\qquad B=\rho T,
\]
where $\rho$ represents the average available resource per round at the experiment scale.
(Importantly, PD-LinUCB uses the dynamic spending target
$b_t=\frac{B-\mathrm{spent}_{t-1}}{T-t+1}$ rather than the fixed ratio $B/T$.)

The following evaluation metrics are reported:
\begin{itemize}[leftmargin=1.2em]
\item \textbf{Total reward:} $R=\sum_{t=1}^{\tau} r_t$;
\item \textbf{Budget utilization:} $\mathrm{spent}/B$;
\item \textbf{Stopping time:} $\tau$ (number of executed actions).
\end{itemize}

\paragraph{Confidence intervals.}
For each metric $m$, using $N$ independent seeds, we compute the mean $\bar m$ and the $95\%$ confidence interval
\[
\bar m \pm t_{0.975,\,N-1}\cdot \frac{s_m}{\sqrt{N}},
\]
where $s_m$ is the sample standard deviation and $t_{0.975,\,N-1}$ is the corresponding quantile of the Student's $t$ distribution.

\subsection{Comparison of Methods in a Fixed Budget Regime}
\label{subsec:main_compare}

We first compare four baseline methods: LinUCB, PD-LinUCB, CostNormUCB[ratio], and the context-independent CF-PD-BwK.
Table~\ref{tab:main_ci_T5000} reports results for $T=5000$ and $\rho=0.7$ under the stop-at-budget protocol, averaged over $N=10$ seeds with $95\%$ confidence intervals.

\begin{table}[t]
\centering
\caption{Results for $T=5000$ and $\rho=0.7$ (stop-at-budget). Mean values over $N=10$ seeds with $95\%$ confidence intervals.}
\label{tab:main_ci_T5000}
\inputifexists{paper_artifacts/tables/main_ci.tex}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.75\linewidth]{paper_artifacts/figures/baselines_cum_reward_full4_arm.png}
\caption{Cumulative reward under the stop-at-budget protocol (example run).}
\label{fig:cum_reward}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.75\linewidth]{paper_artifacts/figures/baselines_cum_cost_full4_arm.png}
\caption{Cumulative cost and budget $B$ (example run).}
\label{fig:cum_cost}
\end{figure}
\subsection{Gamma Sweep for CostNormUCB[sub]}
\label{subsec:gamma_sweep}

For the $\mathrm{UCB}-\gamma c$ heuristic, we perform a sweep over $\gamma$ with fixed $T$ and $\rho$. In the updated evaluation protocol, $\gamma$ is selected by \emph{nested tuning}: we tune on a separate split/seed set and report the selected $\gamma^\star$ only on held-out evaluation seeds.
Since a fixed penalty may cause the method to underutilize the available budget, we report both total reward and budget utilization across the $\gamma$ sweep.

\begin{figure}[t]
\centering

\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth,height=0.28\textheight,keepaspectratio]{paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_reward.png}
  \caption{Total reward (mean $\pm$ 95\% CI).}
  \label{fig:gamma_reward}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth,height=0.28\textheight,keepaspectratio]{paper_artifacts/figures/gamma_sweep_rho0.7_T5000/gamma_sweep_spent.png}
  \caption{Budget utilization: $\mathrm{spent}/B$ (mean $\pm$ 95\% CI).}
  \label{fig:gamma_spent}
\end{subfigure}

\caption{Sweep over the penalty $\gamma$ for the $\mathrm{UCB}-\gamma c$ heuristic (CostNormUCB[sub]) with a fixed budget regime $\rho=0.7$ and horizon $T=5000$ under the stop-at-budget protocol. Results are averaged over $N=10$ seeds and shown with $95\%$ confidence intervals. The x-axis is rendered on a symlog scale so that $\gamma=0$ is included.}
\label{fig:gamma_sweep}
\end{figure}

\subsection{Delay vs.\ No-Delay}
\label{subsec:delay_ablation}

To assess the impact of delayed feedback, we compare two regimes:  
(i) empirical delays with observation-window censoring and  
(ii) the \texttt{no-delay} ablation ($D_t\equiv 0$).

We measure changes in total reward and reward variance, and also record the effect on stopping time $\tau$ and budget utilization $\mathrm{spent}/B$.

\noindent\textbf{No-delay mode.}
The \texttt{no-delay} ablation is implemented by forcing all delays to zero:
$censor\_steps=0$ and $\texttt{delays\_pos}=\{0\}$, so the environment always returns $D_t\equiv 0$
for both $r=1$ and $r=0$.

\begin{table}[t]
\centering
\caption{Delay ablation: empirical delays with censoring versus \texttt{no-delay} ($D_t\equiv 0$).}
\label{tab:delay_ablation}
\inputifexists{paper_artifacts/tables/delay_ablation.tex}
\end{table}

\subsection{Budget Sweep $\rho=B/T$}
\label{subsec:budget_sweep}

The key experiment performs a sweep over the budget ratio $\rho$ for a fixed horizon $T$.
We compare PD-LinUCB with CostNormUCB[sub] (using the best $\gamma$ selected by nested tuning) and show that
the optimal penalty $\gamma^\star$ depends strongly on $\rho$, whereas PD-LinUCB adapts automatically through its dynamic multiplier.
The tuning step uses a disjoint set of seeds and, when a temporal split is available, tunes on training contexts and evaluates on test contexts.

\begin{table}[t]
\centering
\caption{Budget sweep over $\rho=B/T$: PD-LinUCB versus the tuned CostNormUCB[sub] baseline selected by nested tuning.}
\label{tab:budget_sweep}
\inputifexists{paper_artifacts/tables/budget_sweep.tex}
\end{table}

\begin{figure}[t]
\centering
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth,height=0.28\textheight,keepaspectratio]{paper_artifacts/figures/budget_sweep_reward.png}
  \caption{Held-out reward versus $\rho$.}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.49\linewidth}
  \centering
  \includegraphics[width=\linewidth,height=0.28\textheight,keepaspectratio]{paper_artifacts/figures/budget_sweep_gamma_star.png}
  \caption{Selected $\gamma^\star(\rho)$ from nested tuning.}
\end{subfigure}
\caption{Budget sweep with nested tuning: the tuned fixed-penalty baseline remains regime-dependent, while PD-LinUCB does not require manual retuning across budget levels.}
\label{fig:budget_sweep}
\end{figure}

\subsection{Reproducibility: Hyperparameters and Settings}
\label{subsec:hyperparams}

\begin{table}[H]
\centering
\caption{Hyperparameters and experimental settings (default values).}
\label{tab:hyperparams}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Number of arms $K$ & $50$ \\
Context dimension $d$ & $65$ ($64$ hashed categorical + $1$ numerical feature) \\
Horizon $T$ & $5000$ \\
Budget ratio $\rho=B/T$ & $\{0.40, 0.55, 0.70, 0.85\}$ \\
Regularization $\lambda$ (ridge) & $1.0$ \\
LinUCB $\alpha_{\text{lin}}$ & $1.0$ \\
PD-LinUCB $\alpha_{\text{pd}}$ & $1.5$ \\
PD-LinUCB step $\eta$ & $0.05$ \\
CostNormUCB[ratio] $\varepsilon$ & $10^{-3}$ \\
CostNormUCB[sub] $\gamma$ & sweep (e.g., $\{0,0.1,0.3,1,2,3,5,10\}$) \\
Evaluation seeds & $123+s$, $s\in\{1,\dots,N\}$ (e.g., $N=10$) \\
Tuning seeds for $\gamma^\star$ & $10123+s$, $s\in\{1,\dots,N_{\mathrm{tune}}\}$ \\
Context split & test for evaluation; train for nested tuning when available \\
Sampling step $\Delta$ & $3600$ seconds (1 hour) \\
Observation window $W$ & $18{,}000{,}000$ sec ($\approx 208.3$ days, $W/\Delta=5000$) \\
$D_{\max}$ & $5000$ steps \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Prototype Outputs}
\label{subsec:proto_outputs}

After preprocessing the dataset, the following files are stored in the memmap directory:
\texttt{X.npy}, \texttt{A.npy}, \texttt{R.npy}, \texttt{C.npy}, \texttt{D.npy}, \texttt{split.npy}, and \texttt{meta\_and\_stats.npz},
as well as the auxiliary arrays \texttt{costs\_by\_arm.npy}, \texttt{delays\_pos.npy}, and \texttt{delays\_pos\_by\_arm.npz}. Optional files \texttt{arm\_ridge\_stats.npz} and \texttt{arm\_logistic\_stats.npz} store explicit per-arm reward-model parameters fit on the train split.

The experiment scripts produce CSV logs and generate plots for the dependence on $\rho$ and $\gamma$, cumulative reward/cost curves, delay ablations, and held-out simulator diagnostics, including calibration curves, train-vs-test delay CDFs, and arm-level train-vs-test stability checks for costs and delays.
\section{Discussion}
\label{sec:discussion}

Our results can be naturally interpreted through the Lagrangian (primal--dual) perspective of constrained online learning. The budget constraint imposes a long-run restriction on average resource consumption, while the Lagrange multiplier acts as the shadow price of the resource, influencing the action-selection policy. This formulation and the associated primal--dual schemes are standard tools in online convex optimization with long-term constraints, where the dual variable is updated based on the observed constraint violation \cite{mahdavi2012trading,rivera2025online_saddle_point_knapsacks}. 

In the Bandits with Knapsacks (BwK) framework, the same principle underlies many algorithms: the resource is modeled as a knapsack constraint, and online adjustment of the resource ``price'' balances reward and cost \cite{badanidiyuru2013bwk,badanidiyuru2018bwk,li2021primaldual}. In contextual extensions (CBwK/CBwLC), the main challenge is that the algorithm competes with a class of policies while simultaneously controlling resource consumption, often in resource-exhaustion stopping settings. As a result, primal--dual or Lagrangian approaches are typically combined with reduction or oracle-based techniques and careful constraint handling \cite{agrawal2016cbwk,slivkins2024cbwlc,sarkar2026reduction}. Our work intersects these lines while additionally incorporating \emph{delayed reward feedback}, which is known to complicate both analysis and algorithm design in bandit problems \cite{vernade2017conversions}.

\subsection{Why the fixed penalty $\gamma$ is unstable and what PD-LinUCB changes}

The heuristic $\mathrm{UCB}-\gamma c$ corresponds to a static Lagrangian multiplier $\lambda\equiv\gamma$: the resource ``price'' is fixed in advance, and the algorithm behaves as if the reward--cost trade-off were known and stationary. However, in constrained problems the optimal multiplier depends on the \emph{resource regime} and on the statistics of the environment. As the available budget, reward distribution, or delay pattern changes, the appropriate shadow price also changes, and the same value of $\gamma$ may become suboptimal. This observation is consistent with the general role of dual variables in constrained online learning \cite{mahdavi2012trading} and with the BwK intuition that the resource price reflects its scarcity \cite{badanidiyuru2018bwk}.

Empirically, this manifests as strong sensitivity to the choice of $\gamma$: sweeping $\gamma$ changes the total reward and leads to different behavioral regimes. Importantly, under the stop-at-budget protocol the heuristic may achieve higher reward either by allocating resources more effectively across arms or by entering an overly conservative regime that underutilizes the available budget. This ambiguity complicates comparisons based solely on the total reward metric.

PD-LinUCB addresses this instability by replacing the fixed penalty with a dynamic dual variable $\lambda_t$. The update
\[
\lambda_{t+1}=\big[\lambda_t+\eta\,(c(a_t)-b_t)\big]_+,
\qquad
b_t=\frac{B-\mathrm{spent}_{t-1}}{T-t+1},
\]
implements an online adjustment of the resource price based on deviations from the \emph{target consumption rate} (the remaining budget divided by the remaining horizon). This form of the target $b_t$ is particularly suitable for the finite-horizon stop-at-budget protocol: it stabilizes the spending trajectory relative to the remaining resource and prevents premature budget exhaustion without requiring manual tuning of $\gamma$. Conceptually, this mechanism is closely related to primal--dual schemes in constrained online learning \cite{mahdavi2012trading} and to primal--dual algorithms for BwK, where the dual variable represents the adaptive price of the knapsack resource \cite{li2021primaldual}. 

In our contextual setting, the dual update interacts with the UCB reward estimate while preserving interpretability: as the resource becomes scarcer relative to the remaining horizon, the multiplier $\lambda_t$ increases, automatically imposing a stronger penalty on expensive actions.
\subsection{Delays: Why Budget Control Remains Stable While Learning Slows}

Delayed feedback affects two distinct mechanisms in the system:  
(i) resource control and budget stopping rely on immediately observed costs, and  
(ii) reward model learning depends on delayed observations $r_t$.  

As a result, the main degradation caused by delays typically appears through increased uncertainty in reward estimation and a smaller effective sample of observed rewards during the early stages of learning. Classical work on delayed bandits shows that delays require careful treatment when constructing confidence bounds or analyzing regret \cite{vernade2017conversions}. 

In our implementation, we follow the standard approach for linear models with delayed feedback: the \emph{design-now, reward-later} scheme. Specifically, the design matrix $A_a$ is updated immediately after an action is selected (since the context is known instantly), while the response vector $b_a$ is updated only when the corresponding reward arrives. This separation preserves the geometry of contexts independently of the delayed reward signal and provides a natural implementation of delayed feedback for linear UCB models.

It is also important to note that delays affect the compared methods differently. Algorithms that rely heavily on accurate reward estimation (e.g., standard LinUCB) tend to suffer more from delayed updates of $b_a$. In contrast, methods with adaptive resource control may appear more robust because budget management operates independently of reward arrivals. This explains why PD-LinUCB maintains its advantage under delayed feedback in our experiments: the dual mechanism prevents inefficient early spending while the reward model is still being learned.

\subsection{Why a Context-Free Baseline Is Needed}

We include the context-free PD-BwK baseline as a \emph{diagnostic control}, rather than as a direct competitor to contextual methods. Its purpose is to distinguish whether performance improvements arise primarily from (i) effective budget management via the primal--dual mechanism or (ii) the use of contextual information for more accurate arm selection.

If the context-free baseline performs similarly to PD-LinUCB, this indicates that much of the improvement is driven by differences between arms at the average reward--cost level, and that contextual information provides limited additional benefit in the given horizon and delay regime. Conversely, if PD-LinUCB significantly outperforms the context-free baseline, this implies that contextual information meaningfully changes the relative ordering of actions and yields additional value beyond budget control. This diagnostic aligns with the intuition of BwK problems, where the value of an action relative to a resource depends on both the reward distribution and the cost structure \cite{badanidiyuru2018bwk,li2021primaldual}.

\subsection{Limitations and Future Directions}

Although our experiments use real cost and conversion timestamp fields and a large-scale prototype built on the full Criteo log with a temporal train/test split, the current framework remains a simulator: rewards are generated from a calibrated arm-specific model, and the online cost is aggregated to an arm-dependent value. 

Several extensions are particularly promising. First, many practical applications involve context-dependent costs $c_t(a,x_t)$ (e.g., in advertising auctions). Second, more general resource constraints may arise, including multiple simultaneous resources (the CBwLC setting) with vector-valued dual variables \cite{slivkins2023contextual,slivkins2024cbwlc}. Third, a more expressive reward model such as GLM-UCB could better reflect the Bernoulli nature of the response. Finally, non-stationary environments—where delays interact with distribution drift—represent an important direction for future work. In all these cases, the same central principle remains: resource prices should adapt online, while delayed feedback requires a clear separation between instantaneous information (context and cost) and delayed signals (rewards).

\paragraph{Summary}

In practical settings with a global budget and delayed feedback, two loops must be handled carefully: online resource control (cost accounting together with the stop-at-budget rule) and reward learning from delayed observations (the design-now, reward-later scheme). Under this perspective, the fixed penalty heuristic $\mathrm{UCB}-\gamma c$ is fragile because it requires manual tuning to the resource regime, whereas the dynamic primal--dual multiplier $\lambda_t$ adaptively adjusts the resource price and stabilizes performance while remaining compatible with delayed updates \cite{mahdavi2012trading,li2021primaldual,vernade2017conversions}.

\section{Conclusion}
\label{sec:conclusion}

In this paper, we studied a contextual bandit problem with a budget constraint and delayed feedback under the stop-at-budget protocol, where costs are incurred immediately while rewards may be observed with delay. We proposed a simple and reproducible algorithm, PD-LinUCB --- a variant of Disjoint LinUCB with Lagrangian (primal--dual) budget control implemented via a dynamic dual variable $\lambda_t$ and delayed-feedback semantics based on the design-now, reward-later update scheme. For empirical evaluation, we developed a scalable semi-synthetic benchmark built on the full Criteo Attribution dataset (16,468,027 records), stored in memmap format and split temporally into train and test portions, using real cost and conversion timestamp fields to model both resource consumption and delays, including observation-window censoring.

Our experiments demonstrate that the adaptive primal--dual mechanism provides robust performance improvements compared with both cost-agnostic LinUCB and simple cost-aware heuristics. In particular, PD-LinUCB consistently achieves higher total reward while maintaining comparable budget utilization under the stop-at-budget protocol. In contrast, the fixed-penalty heuristic $\mathrm{UCB}-\gamma c$ shows strong sensitivity to the choice of $\gamma$: the optimal value $\gamma^\star$ depends on the budget regime and may push the method into different behavioral regimes, sometimes leading to budget underutilization. This observation motivates evaluating algorithms jointly in terms of total reward and budget utilization ($\mathrm{spent}/B$), rather than relying on a single metric. Overall, the results indicate that the adaptive multiplier $\lambda_t$ provides a practical alternative to manually tuning a fixed penalty, stabilizing the cost trajectory and overall performance under delayed feedback.

The current study has several limitations. The online environment is a simulator with fixed arm-specific reward parameters, and the baseline cost model assumes that costs depend only on the chosen action. Several extensions are therefore natural directions for future work. These include multi-resource constraints (vector-valued budgets and multivariate dual variables), context-dependent costs, and non-stationary environments with distributional drift. Additional baselines, such as constrained Thompson Sampling, would also provide useful comparisons, as would a more rigorous theoretical analysis of primal--dual UCB under delayed feedback \cite{moghimi2025neural,deb2025_thompson_constrained_bandits}. Nevertheless, even in its current form, the proposed prototype demonstrates that primal--dual budget control provides a natural and practically robust approach for constrained online learning with delayed feedback, where reward learning and resource management must be handled simultaneously.
\appendix
\section{Pseudocodes}
\label{app:pseudocode}

\subsection{General Pending Processing Mechanism (Pending Queue)}
\label{app:pending}

We use a unified deferred-update scheduling mechanism. At each round $t$, all updates scheduled for time $t$ are first applied. The algorithm then selects an action, and any new update is scheduled for time $t+D_t$.

In \emph{design-now, reward-later} schemes, only updates that require the reward (i.e., updates of vectors $b_a$) are queued, while the design matrices $A_a$ are updated immediately when an action is selected.

\begin{center}
\begin{tabular}{p{0.95\linewidth}}
\toprule
\textbf{Procedure: Pending Queue (updates of $b$ only)} \\
\midrule
\textbf{Initialization:} \texttt{pending} $\leftarrow$ dictionary with key $t_{\text{due}}$ and value a list of triples $(a,x,r)$.\\[2pt]

\textbf{For} $t=1,\dots,T$:
\begin{enumerate}[leftmargin=1.2em]
\item \textbf{Apply pending updates:} if $t \in \texttt{pending}$, then for each $(a,x,r)\in\texttt{pending}[t]$, execute
\[
b_a \leftarrow b_a + xr .
\]

\item The algorithm selects $a_t$, immediately updates $A_{a_t}$ (and $A^{-1}_{a_t}$), receives $(r_t,c_t,D_t)$, and schedules the reward update at time $t+D_t$ (or performs it immediately if $D_t=0$).
\end{enumerate}
\\
\bottomrule
\end{tabular}
\end{center}

\subsection{Disjoint LinUCB with Delayed Feedback}
\label{app:linucb}

\begin{center}
\begin{tabular}{p{0.95\linewidth}}
\toprule
\textbf{Algorithm 1: Disjoint LinUCB with Delayed Feedback (design-now, reward-later) + feasible-set stop-at-budget} \\
\midrule

\textbf{Parameters:} number of arms $K$, context dimension $d$, regularization $\lambda>0$, UCB parameter $\alpha>0$, budget $B>0$, horizon $T$.\\

\textbf{Initialization:} for all $a\in\{1,\dots,K\}$:
\[
A_a \leftarrow \lambda I_d,\qquad
A_a^{-1}\leftarrow \lambda^{-1}I_d,\qquad
b_a \leftarrow 0_d.
\]

$\mathrm{spent}_0 \leftarrow 0$. The \texttt{pending} queue is empty.\\

\textbf{For} $t=1,\dots,T$:

\begin{enumerate}[leftmargin=1.2em]

\item \textbf{Apply pending updates:}  
for each $(a,x,r)$ in \texttt{pending[$t$]}, execute  
\[
b_a \leftarrow b_a + xr .
\]

\item \textbf{Feasible set (stop-at-budget):}
\[
\mathcal{A}_t =
\{a\in\mathcal{A}: \mathrm{spent}_{t-1}+c(a)\le B\}.
\]

If $\mathcal{A}_t=\varnothing$, stop and set $\tau=t-1$.

\item \textbf{Compute UCB indices:}

For each $a\in\mathcal{A}_t$:

\[
\widehat{\theta}_a \leftarrow A_a^{-1} b_a
\]

\[
\mathrm{UCB}_t(x_t,a) =
\widehat{\theta}_a^\top x_t +
\alpha\sqrt{x_t^\top A_a^{-1}x_t}.
\]

\item \textbf{Action selection:}
\[
a_t \leftarrow
\arg\max_{a\in\mathcal{A}_t}
\mathrm{UCB}_t(x_t,a).
\]

\item \textbf{Commit cost:}

\[
\mathrm{spent}_t
\leftarrow
\mathrm{spent}_{t-1}+c(a_t).
\]

\item \textbf{Immediate design update:}

\[
A_{a_t}
\leftarrow
A_{a_t}+x_tx_t^\top.
\]

Update $A^{-1}_{a_t}$ using Sherman--Morrison:

\[
A_{a_t}^{-1}\leftarrow
A_{a_t}^{-1}
-
\frac{
A_{a_t}^{-1}x_tx_t^\top A_{a_t}^{-1}
}{
1+x_t^\top A_{a_t}^{-1}x_t
}.
\]

\item Receive $(r_t,\tilde c_t,D_t)$ from the environment.

\item \textbf{Reward update:}

If $D_t=0$:

\[
b_{a_t}\leftarrow b_{a_t}+x_tr_t
\]

otherwise add $(a_t,x_t,r_t)$ to \texttt{pending[$t+D_t$]}.

\end{enumerate}
\\
\bottomrule
\end{tabular}
\end{center}

\subsection{Primal--Dual LinUCB (PD-LinUCB) with Budget and Delays}
\label{app:pdlinucb}

\begin{center}
\begin{tabular}{p{0.95\linewidth}}
\toprule
\textbf{Algorithm 2: Primal--Dual Disjoint LinUCB with Budget and Delays} \\
\textbf{(design-now, reward-later + feasible-set stop-at-budget)} \\
\midrule

\textbf{Parameters:} number of arms $K$, dimension $d$, regularization $\lambda>0$, UCB parameter $\alpha>0$, step size $\eta>0$, budget $B>0$, horizon $T$.\\

\textbf{Initialization:}

For all $a\in\{1,\dots,K\}$:

\[
A_a \leftarrow \lambda I_d,
\quad
A_a^{-1}\leftarrow \lambda^{-1}I_d,
\quad
b_a \leftarrow 0_d.
\]

\[
\lambda_1 \leftarrow 0,
\qquad
\mathrm{spent}_0 \leftarrow 0.
\]

The \texttt{pending} queue is empty.\\

\textbf{For} $t=1,\dots,T$:

\begin{enumerate}[leftmargin=1.2em]

\item \textbf{Apply pending updates:}

for each $(a,x,r)$ in \texttt{pending[$t$]}:

\[
b_a \leftarrow b_a + xr.
\]

\item \textbf{Feasible set (stop-at-budget):}

\[
\mathcal{A}_t =
\{a\in\mathcal{A}:\mathrm{spent}_{t-1}+c(a)\le B\}.
\]

If $\mathcal{A}_t=\varnothing$, stop and set $\tau=t-1$.

\item \textbf{Compute indices:}

For each $a\in\mathcal{A}_t$:

\[
\widehat{\theta}_a \leftarrow A_a^{-1} b_a
\]

\[
\mathrm{UCB}_t(x_t,a)
=
\widehat{\theta}_a^\top x_t
+
\alpha\sqrt{x_t^\top A_a^{-1}x_t}.
\]

\item \textbf{Primal action selection:}

\[
a_t
\leftarrow
\arg\max_{a\in\mathcal{A}_t}
\Big(
\mathrm{UCB}_t(x_t,a)
-
\lambda_t c(a)
\Big).
\]

\item \textbf{Dual update (remaining-budget target):}

\[
b_t
=
\frac{B-\mathrm{spent}_{t-1}}{T-t+1}
\]

\[
\lambda_{t+1}
\leftarrow
\max\{0,\lambda_t+\eta(c(a_t)-b_t)\}.
\]

\item \textbf{Commit cost:}

\[
\mathrm{spent}_t
\leftarrow
\mathrm{spent}_{t-1}+c(a_t).
\]

\item \textbf{Immediate design update:}

\[
A_{a_t}\leftarrow A_{a_t}+x_tx_t^\top
\]

and update $A^{-1}_{a_t}$ via Sherman--Morrison.

\item Receive $(r_t,\tilde c_t,D_t)$ from the environment.

\item \textbf{Reward update:}

If $D_t=0$:

\[
b_{a_t}\leftarrow b_{a_t}+x_tr_t
\]

otherwise add $(a_t,x_t,r_t)$ to \texttt{pending[$t+D_t$]}.

\end{enumerate}
\\
\bottomrule
\end{tabular}
\end{center}
\subsection{Cost-Aware Heuristics}
\label{app:cost_heuristics}

\paragraph{CostNormUCB[ratio] (heuristic $\mathrm{UCB}/(c+\varepsilon)$).}

\begin{center}
\begin{tabular}{p{0.95\linewidth}}
\toprule
\textbf{Algorithm 3: CostNormUCB[ratio] + feasible-set stop-at-budget} \\
\midrule
\textbf{Parameters:} $K,d,\lambda,\alpha$, $\varepsilon>0$, budget $B$, horizon $T$.\\
\textbf{Initialization:} same as in Algorithm~1.\\

\textbf{For} $t=1,\dots,T$:
\begin{enumerate}[leftmargin=1.2em]

\item Apply pending updates as in Algorithm~1 (updates of $b_a$).

\item \textbf{Feasible set (stop-at-budget):}
\[
\mathcal{A}_t=
\{a\in\mathcal{A}: \mathrm{spent}_{t-1}+c(a)\le B\}.
\]
If $\mathcal{A}_t=\varnothing$, stop and set $\tau=t-1$.

\item For each arm $a\in\mathcal{A}_t$, compute $\mathrm{UCB}_t(x_t,a)$ as in~\eqref{eq:ucb}.

\item \textbf{Action selection:}
\[
a_t \leftarrow 
\arg\max_{a\in\mathcal{A}_t}
\frac{\mathrm{UCB}_t(x_t,a)}{c(a)+\varepsilon}.
\]

\item \textbf{Commit cost:}
\[
\mathrm{spent}_t\leftarrow \mathrm{spent}_{t-1}+c(a_t).
\]

\item Immediately update $A_{a_t}$ (as in Algorithm~1), then receive $(r_t,c_t,D_t)$ from the environment and update $b_{a_t}$: immediately if $D_t=0$, otherwise schedule the update in \texttt{pending[$t+D_t$]}.

\end{enumerate}
\\
\bottomrule
\end{tabular}
\end{center}

\paragraph{CostNormUCB[sub] (heuristic $\mathrm{UCB}-\gamma c$).}

\begin{center}
\begin{tabular}{p{0.95\linewidth}}
\toprule
\textbf{Algorithm 4: CostNormUCB[sub] + feasible-set stop-at-budget} \\
\midrule
\textbf{Parameters:} $K,d,\lambda,\alpha$, fixed penalty $\gamma\ge 0$, budget $B$, horizon $T$.\\
\textbf{Initialization:} same as in Algorithm~1.\\

\textbf{For} $t=1,\dots,T$:
\begin{enumerate}[leftmargin=1.2em]

\item Apply pending updates as in Algorithm~1 (updates of $b_a$).

\item \textbf{Feasible set (stop-at-budget):}
\[
\mathcal{A}_t=
\{a\in\mathcal{A}: \mathrm{spent}_{t-1}+c(a)\le B\}.
\]
If $\mathcal{A}_t=\varnothing$, stop and set $\tau=t-1$.

\item For each arm $a\in\mathcal{A}_t$, compute $\mathrm{UCB}_t(x_t,a)$ as in~\eqref{eq:ucb}.

\item \textbf{Action selection:}
\[
a_t \leftarrow 
\arg\max_{a\in\mathcal{A}_t}
\big(\mathrm{UCB}_t(x_t,a)-\gamma\,c(a)\big).
\]

\item \textbf{Commit cost:}
\[
\mathrm{spent}_t\leftarrow \mathrm{spent}_{t-1}+c(a_t).
\]

\item Immediately update $A_{a_t}$ (as in Algorithm~1), then receive $(r_t,c_t,D_t)$ and update $b_{a_t}$: immediately if $D_t=0$, otherwise schedule the update in \texttt{pending[$t+D_t$]}.

\end{enumerate}
\\
\bottomrule
\end{tabular}
\end{center}

\subsection{Context-Independent Primal--Dual BwK (CF-PD-BwK)}
\label{app:cf_pdbwk}

\begin{center}
\begin{tabular}{p{0.95\linewidth}}
\toprule
\textbf{Algorithm 5: Context-Free Primal--Dual BwK with Delays} \\
\textbf{(feasible-set stop-at-budget + pending queue)} \\
\midrule

\textbf{Parameters:} number of arms $K$, UCB parameter $\alpha>0$, step size $\eta>0$, budget $B>0$, horizon $T$.\\

\textbf{Initialization:} for all $a$:
\[
n_a\leftarrow 0,\qquad s_a\leftarrow 0.
\]

\[
\lambda_1\leftarrow 0,\qquad
\mathrm{spent}_0\leftarrow 0.
\]

The \texttt{pending} queue is empty.\\

\textbf{For} $t=1,\dots,T$:

\begin{enumerate}[leftmargin=1.2em]

\item \textbf{Apply pending updates:}  

for each $(a,r)$ arriving in \texttt{pending[$t$]} update

\[
n_a\leftarrow n_a+1,\qquad
s_a\leftarrow s_a+r.
\]

\item Compute for each arm $a$:

\[
\widehat{\mu}_a
\leftarrow
\frac{s_a}{\max(1,n_a)},
\]

\[
\mathrm{UCB}_t(a)
\leftarrow
\widehat{\mu}_a
+
\alpha
\sqrt{
\frac{\log(t+1)}{\max(1,n_a)}
}.
\]

\item \textbf{Feasible set (stop-at-budget):}

\[
\mathcal{A}_t
=
\{a\in\mathcal{A}:
\mathrm{spent}_{t-1}+c(a)\le B\}.
\]

If $\mathcal{A}_t=\varnothing$, stop and set $\tau=t-1$.

\item \textbf{Primal action selection:}

\[
a_t
\leftarrow
\arg\max_{a\in\mathcal{A}_t}
\big(\mathrm{UCB}_t(a)-\lambda_t c(a)\big).
\]

\item \textbf{Dual update (remaining-budget target):}

\[
b_t
=
\frac{B-\mathrm{spent}_{t-1}}{T-t+1},
\]

\[
\lambda_{t+1}
\leftarrow
\max\{0,\lambda_t+\eta(c(a_t)-b_t)\}.
\]

\item \textbf{Commit cost:}

\[
\mathrm{spent}_t
\leftarrow
\mathrm{spent}_{t-1}+c(a_t).
\]

\item Receive $(r_t,\tilde c_t,D_t)$ from the environment (using $c(a_t)$ for the budget).

\item \textbf{Reward update:}

If $D_t=0$ update $(n_{a_t},s_{a_t})$ immediately, otherwise add $(a_t,r_t)$ to \texttt{pending[$t+D_t$]}.

\end{enumerate}
\\
\bottomrule
\end{tabular}
\end{center}
\end{document}
