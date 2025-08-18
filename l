import numpy as np
import pandas as pd

def build_13_pools_minA(
    df: pd.DataFrame,
    score_col: str = 'score_NM',
    target_col: str | None = 'target',
    A_min_share: float = 0.05,    # минимум для пула A
    round_to: float = 1.0,        # округление краёв
    ensure_min_gap: float | None = None,  # например, 7.0 очков; если None — не форсируем
    pool_names = ('M','L','K','J','I','H','G','F','E','D','C','B','A'),
):
    """
    Делает 13 пулов так, чтобы пул A был топ-квантилью с долей >= A_min_share.
    Остальные 12 пулов строятся как квантильные на нижних (1 - A_min_share).
    По желанию можно потребовать минимальный зазор между соседними порогами (ensure_min_gap).
    """
    assert len(pool_names) == 13, "Нужно ровно 13 имён пулов"

    s = df[score_col].astype(float).to_numpy()
    n = len(s)
    if n < 13:
        raise ValueError("Слишком мало наблюдений для 13 пулов")

    # 1) Край для входа в A — по квантилю (например, 95-й)
    qA = 1.0 - float(A_min_share)
    qA = min(max(qA, 0.0), 1.0)
    edge_A = np.quantile(s, qA)

    # 2) Нижнюю часть [min, edge_A] режем на 12 квантилей
    q_lower = np.linspace(0, qA, 13)  # 13 краёв → 12 интервалов
    edges_lower = np.quantile(s, q_lower)
    edges = np.r_[edges_lower, np.quantile(s, 1.0)]  # добавляем максимум как последний край

    # 3) Снэп к «красивой» сетке
    if round_to and round_to > 0:
        edges = np.round(edges / round_to) * round_to

    # 4) (опционально) форсируем минимальный зазор между соседними границами
    if ensure_min_gap is not None and ensure_min_gap > 0:
        gaps = np.diff(edges)
        # прямой проход
        for i in range(len(gaps)):
            need = edges[i] + ensure_min_gap
            if edges[i+1] < need:
                edges[i+1] = need
        # обратный проход
        for i in range(len(gaps)-1, -1, -1):
            need = edges[i+1] - ensure_min_gap
            if edges[i] > need:
                edges[i] = need
        # после подвижек повторно закруглим и починим монотонность
        if round_to and round_to > 0:
            edges = np.round(edges / round_to) * round_to
            for i in range(len(edges)-1):
                if edges[i+1] <= edges[i]:
                    edges[i+1] = edges[i] + ensure_min_gap

    # 5) Назначаем метки
    out = df.copy()
    out['POOL'] = pd.cut(
        out[score_col].astype(float),
        bins=edges,
        labels=pool_names,  # низкие → 'M' ... высокие → 'A'
        include_lowest=True,
        right=True
    )

    # 6) Отчёт
    grp = out.groupby('POOL', observed=True)
    agg = {'n': ('POOL','size'),
           'min_score': (score_col,'min'),
           'max_score': (score_col,'max')}
    if target_col and target_col in out.columns:
        agg['defaults'] = (target_col, 'sum')

    rep = grp.agg(**agg).reset_index()
    rep['share'] = rep['n'] / len(out)
    if 'defaults' in rep.columns:
        rep['rate'] = rep['defaults'] / rep['n']

    info = {
        'edges': edges.tolist(),
        'gaps': np.diff(edges).tolist(),
        'A_share_actual': float(rep.loc[rep['POOL']=='A','share'].iloc[0]) if 'A' in rep['POOL'].values else np.nan
    }
    return out, rep.sort_values('min_score'), info

# хотим минимум 5% в A, аккуратные края кратные 1 очку, без жёсткого min-gap:
z13, pools_report, info = build_13_pools_minA(
    df=zayavki_data,
    score_col='score_NM',
    target_col='target',
    A_min_share=0.05,
    round_to=1,
    ensure_min_gap=None  # если нужно — поставь 7.0
)
print("Доля A:", round(info['A_share_actual']*100, 2), "%")
print("Края:", [int(e) for e in info['edges']])
print(pools_report[['POOL','n','share','min_score','max_score']])
