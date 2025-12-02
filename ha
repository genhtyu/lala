import numpy as np
import pandas as pd

def pd_metrics(df, bad_col='target1', amt_col='amount'):
    n = len(df)
    if n == 0:
        return 0.0, 0.0, 0.0

    bad_cnt = df[bad_col].sum()
    pd_q = 100 * bad_cnt / n  # PD по штукам, %

    total_amt = df[amt_col].sum()
    bad_amt = df.loc[df[bad_col] == 1, amt_col].sum()
    pd_s = 100 * bad_amt / total_amt if total_amt > 0 else 0.0  # PD по сумме, %

    # PD_T(s) = PD(s)% * PD(q в долях)
    pd_t = pd_s * (pd_q / 100)

    return pd_q, pd_s, pd_t


def split_bucket10_by_target_pdt(
    data,
    score_col='SCORE1',
    bucket_col='bucket',
    last_bucket=10,
    bad_col='target1',
    amt_col='amount',
    targets_pdt=[14, 17, 20, 23],  # целевые PD_T(s), %
    new_bucket_start=11           # номера новых бакетов: 11..15
):
    """
    Делим исходный бакет `last_bucket` на len(targets_pdt)+1 подбакетов так,
    чтобы PD_T(s) по каждому примерно попадал в заданные targets_pdt.

    Возвращает DataFrame с границами скоринга и фактическими PD по новым бакетам.
    """

    # берём только исходный 10-й бакет и сортируем по скору
    df10 = (data[data[bucket_col] == last_bucket]
            .sort_values(score_col)
            .reset_index(drop=True))

    n = len(df10)
    if n == 0:
        raise ValueError("В исходном бакете нет записей")

    cuts = []   # позиции разрезов (по iloc)
    stats = []  # PD_q, PD_s, PD_T для каждого нового бакета

    start = 0
    remaining_splits = len(targets_pdt) + 1  # сколько подбакетов ещё нужно, включая последний

    for t in targets_pdt:
        best_idx = None
        best_diff = np.inf

        # проходимся по возможным границам для текущего подбакета
        # оставляем минимум по 1 наблюдению на каждый оставшийся подбакет
        for end in range(start + 1, n - (remaining_splits - 1) + 1):
            _, _, pdt = pd_metrics(df10.iloc[start:end], bad_col=bad_col, amt_col=amt_col)
            diff = abs(pdt - t)
            if diff < best_diff:
                best_diff = diff
                best_idx = end

        cuts.append(best_idx)
        pd_q, pd_s, pdt = pd_metrics(df10.iloc[start:best_idx], bad_col=bad_col, amt_col=amt_col)
        stats.append((pd_q, pd_s, pdt))
        start = best_idx
        remaining_splits -= 1

    # последний подбакет — всё, что осталось
    pd_q, pd_s, pdt = pd_metrics(df10.iloc[start:], bad_col=bad_col, amt_col=amt_col)
    stats.append((pd_q, pd_s, pdt))

    # переводим позиции разрезов в скоровые границы
    left_edges = [df10[score_col].iloc[0]] + [df10[score_col].iloc[i] for i in cuts]
    right_edges = [df10[score_col].iloc[i-1] for i in cuts] + [df10[score_col].iloc[-1]]

    # собираем итоговую табличку по новым бакетам 11..15
    rows = []
    for i in range(len(stats)):
        pd_q, pd_s, pdt = stats[i]
        rows.append({
            'bucket': new_bucket_start + i,      # 11,12,13,14,15
            'from_score': left_edges[i],
            'to_score': right_edges[i],
            'PD(q), %': pd_q,
            'PD(s), %': pd_s,
            'PD_T(s), %': pdt
        })

    return pd.DataFrame(rows)

new_10_buckets = split_bucket10_by_target_pdt(
    data=data,
    score_col='SCORE1',
    bucket_col='bucket',
    last_bucket=10,
    bad_col='target1',
    amt_col='amount',
    targets_pdt=[14, 17, 20, 23],  # цели для 10-1..10-4
    new_bucket_start=11            # 11..15
)

display(new_10_buckets)
