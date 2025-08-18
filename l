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








import numpy as np
import pandas as pd

def build_13_pools_with_constraints(
    df: pd.DataFrame,
    score_col: str = 'score_NM',
    target_col: str | None = 'target',
    pool_names = ('M','L','K','J','I','H','G','F','E','D','C','B','A'),
    bad_pools = ('M','L','K'),      # какие считаем «плохими»
    bad_max_share: float = 0.10,    # суммарная доля «плохих» пулов ≤ 10%
    A_min_share: float = 0.05,      # A минимум 5%
    round_to: float = 1.0,          # округление краёв к «сетке» (очки)
    ensure_min_gap: float | None = None   # напр., 7.0 (если None — не форсируем)
):
    """
    Делает 13 пулов, соблюдая:
      - суммарная доля bad-пулов ≤ bad_max_share,
      - доля A ≥ A_min_share,
      - середина забирает оставшуюся массу (равные квантильные доли внутри своих зон).
    Число пулов строго 13.
    """

    assert len(pool_names) == 13, "Должно быть ровно 13 имён пулов"
    s = df[score_col].astype(float).to_numpy()
    n = len(s)
    if n < 13:
        raise ValueError("Слишком мало наблюдений для 13 пулов")

    pool_names = tuple(pool_names)
    bad_pools  = tuple(bad_pools)
    # проверка, что bad-пулы — нижние в списке
    assert all(p in pool_names for p in bad_pools), "bad_pools должны входить в pool_names"
    bad_idx = [pool_names.index(p) for p in bad_pools]
    assert bad_idx == sorted(bad_idx) and max(bad_idx) < pool_names.index('A'), "bad-пулы должны быть нижними"

    # считаем, сколько пулов в каждой зоне
    W = len(bad_pools)                  # «плохие»
    top = 1                             # верхний пул A
    mid = 13 - W - top                  # средние

    # проверка осуществимости
    eps = 1e-9
    if A_min_share + bad_max_share > 1 - eps:
        raise ValueError("Сумма A_min_share и bad_max_share слишком велика — не остаётся места для средних пулов.")

    # доли по зонам
    share_bad = float(bad_max_share)
    share_A   = float(A_min_share)
    share_mid = 1.0 - share_bad - share_A

    # квоты на пул внутри каждой зоны
    q_bad_per_pool = share_bad / W if W > 0 else 0.0
    q_mid_per_pool = share_mid / mid if mid > 0 else 0.0

    # строим кумулятивные квоти
    q_edges = [0.0]
    # bad зона
    for _ in range(W):
        q_edges.append(q_edges[-1] + q_bad_per_pool)
    # mid зона
    for _ in range(mid):
        q_edges.append(q_edges[-1] + q_mid_per_pool)
    # A зона — добиваем до 1.0 последним краем
    q_edges[-1] = 1.0  # гарантированно последний край = 1.0
    if len(q_edges) != 14:
        # на всякий случай восстановим длину 14 (0..1 с 13 интервалами)
        q_edges = np.linspace(0, 1, 14).tolist()

    # пересчитываем квантильные края по score
    edges = np.quantile(s, q_edges)

    # округление краёв
    if round_to and round_to > 0:
        edges = np.round(edges / round_to) * round_to

    # опционально: обеспечить минимальный зазор по очкам
    if ensure_min_gap is not None and ensure_min_gap > 0:
        gaps = np.diff(edges)
        # прямой проход
        for i in range(len(gaps)):
            need = edges[i] + ensure_min_gap
            if edges[i+1] < need:
                edges[i+1] = need
        # обратный
        for i in range(len(gaps)-1, -1, -1):
            need = edges[i+1] - ensure_min_gap
            if edges[i] > need:
                edges[i] = need
        # после подвижек: повторное округление и монотонность
        if round_to and round_to > 0:
            edges = np.round(edges / round_to) * round_to
            for i in range(len(edges)-1):
                if edges[i+1] <= edges[i]:
                    edges[i+1] = edges[i] + ensure_min_gap

    # разрезаем на пулы
    out = df.copy()
    out['POOL'] = pd.cut(
        out[score_col].astype(float),
        bins=edges,
        labels=pool_names,   # низкие значения → первые метки списка
        include_lowest=True,
        right=True
    )

    # отчёт
    grp = out.groupby('POOL', observed=True)
    agg = {'n': ('POOL','size'),
           'min_score': (score_col,'min'),
           'max_score': (score_col,'max')}
    if target_col and target_col in out.columns:
        agg['defaults'] = (target_col,'sum')
    rep = grp.agg(**agg).reset_index()
    rep['share'] = rep['n'] / len(out)
    if 'defaults' in rep.columns:
        rep['rate'] = rep['defaults'] / rep['n']

    # проверки ограничений
    actual_A_share = float(rep.loc[rep['POOL']=='A','share'].iloc[0]) if (rep['POOL']=='A').any() else np.nan
    actual_bad_share = float(rep.loc[rep['POOL'].isin(bad_pools),'share'].sum())
    info = {
        'edges': edges.tolist(),
        'gaps': np.diff(edges).tolist(),
        'A_share_actual': actual_A_share,
        'bad_share_actual': actual_bad_share,
        'constraints_ok': (actual_A_share + 1e-9 >= A_min_share) and (actual_bad_share <= bad_max_share + 1e-9)
    }
    return out, rep.sort_values('min_score'), info
