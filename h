def pd_metrics_bucket_style(
    df,
    hdp_def_col='hdp_def',        # дни/факт дефолта для PD(q)
    expos_col='expos',            # экспозиция для PD(q)
    num_s_col='ЧИСЛИТЕЛЬ',        # числитель для PD(s)
    den_s_col='ЗНАМЕНАТЕЛЬ'       # знаменатель для PD(s)
):
    """
    Считает PD(q), PD(s) и PD_T(s) в процентах по кусочку данных df
    по той же формуле, что и при агрегации по бакетам.
    """

    n = len(df)
    if n == 0:
        return 0.0, 0.0, 0.0

    # --- PD(q) по количеству (как в коде для бакетов) ---
    hdp_def_sum = df[hdp_def_col].sum()
    expos_sum = df[expos_col].sum()

    if expos_sum == 0:
        pd_q = 0.0
    else:
        # та же формула: hdp_def / (expos * 1/365) * 100
        pd_q = (hdp_def_sum / (expos_sum * (1/365))) * 100

    # --- PD(s) по сумме (как в коде для бакетов) ---
    num_s_sum = df[num_s_col].sum()
    den_s_sum = df[den_s_col].sum()

    if den_s_sum == 0:
        pd_s = 0.0
    else:
        pd_s = (num_s_sum / (den_s_sum * (1/365))) * 100

    # --- PD_T(s) = PD(s) * PD(q в долях) ---
    pd_t = pd_s * (pd_q / 100.0)

    return pd_q, pd_s, pd_t

import numpy as np
import pandas as pd

def split_bucket10_by_target_pdt(
    data,
    score_col='SCORE1',
    bucket_col='bucket',
    last_bucket=10,
    # названия колонок для PD:
    hdp_def_col='hdp_def',
    expos_col='expos',
    num_s_col='ЧИСЛИТЕЛЬ',
    den_s_col='ЗНАМЕНАТЕЛЬ',
    # целевые PD_T(s) по подбакетам:
    targets_pdt=[14, 17, 20, 23],  # для 10-1..10-4, 10-5 будет "хвост"
    new_bucket_start=11            # новые бакеты: 11,12,13,14,15
):
    """
    Делим исходный бакет `last_bucket` на len(targets_pdt)+1 подбакетов так,
    чтобы PD_T(s) по каждому подбакету был максимально близок заданным targets_pdt.
    PD(q) и PD(s) считаются по той же формуле, что и для агрегированных бакетов.
    """

    # 1) Берём только 10-й бакет и сортируем по скору
    df10 = (data[data[bucket_col] == last_bucket]
            .sort_values(score_col)
            .reset_index(drop=True))

    n = len(df10)
    if n == 0:
        raise ValueError("В исходном бакете нет записей")

    cuts = []   # индексы границ
    stats = []  # (PD_q, PD_s, PD_T) для каждого нового подбакета

    start = 0
    remaining_splits = len(targets_pdt) + 1  # сколько подбакетов ещё нужно, включая последний

    for target in targets_pdt:
        best_idx = None
        best_diff = np.inf

        # оставляем минимум по 1 наблюдению на каждый оставшийся подбакет
        for end in range(start + 1, n - (remaining_splits - 1) + 1):
            _, _, pdt = pd_metrics_bucket_style(
                df10.iloc[start:end],
                hdp_def_col=hdp_def_col,
                expos_col=expos_col,
                num_s_col=num_s_col,
                den_s_col=den_s_col,
            )
            diff = abs(pdt - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = end

        cuts.append(best_idx)
        pd_q, pd_s, pdt = pd_metrics_bucket_style(
            df10.iloc[start:best_idx],
            hdp_def_col=hdp_def_col,
            expos_col=expos_col,
            num_s_col=num_s_col,
            den_s_col=den_s_col,
        )
        stats.append((pd_q, pd_s, pdt))
        start = best_idx
        remaining_splits -= 1

    # последний подбакет — всё, что осталось
    pd_q, pd_s, pdt = pd_metrics_bucket_style(
        df10.iloc[start:],
        hdp_def_col=hdp_def_col,
        expos_col=expos_col,
        num_s_col=num_s_col,
        den_s_col=den_s_col,
    )
    stats.append((pd_q, pd_s, pdt))

    # 3) Переводим разрезы в границы скорбаллов
    left_edges = [df10[score_col].iloc[0]] + [df10[score_col].iloc[i] for i in cuts]
    right_edges = [df10[score_col].iloc[i-1] for i in cuts] + [df10[score_col].iloc[-1]]

    # 4) Собираем итоговую таблицу: новые бакеты 11..15
    rows = []
    for i, (pd_q, pd_s, pdt) in enumerate(stats):
        rows.append({
            'bucket': new_bucket_start + i,      # 11,12,13,14,15

'from_score': left_edges[i],
            'to_score': right_edges[i],
            'PD(q), %': round(pd_q, 3),
            'PD(s), %': round(pd_s, 3),
            'PD_T(s), %': round(pdt, 3),
        })

    return pd.DataFrame(rows)

new_10_buckets = split_bucket10_by_target_pdt(
    data=data,                # твой полный датафрейм с клиентами
    score_col='SCORE1',
    bucket_col='bucket',
    last_bucket=10,
    hdp_def_col='hdp_def',
    expos_col='expos',
    num_s_col='ЧИСЛИТЕЛЬ',    # подставь реальные названия
    den_s_col='ЗНАМЕНАТЕЛЬ',
    targets_pdt=[14, 17, 20, 23],
    new_bucket_start=11
)

display(new_10_buckets)





