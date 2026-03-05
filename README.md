# paper_cbwk_delays — Contextual BwK/CBwK with delayed feedback (PD-LinUCB) on Criteo Attribution

Прототип к статье про **контекстные бандиты с бюджетным ограничением (BwK/CBwK)** и **задержанным фидбеком**.  
Фокус: воспроизводимая эмпирика на полном датасете **Criteo Attribution** с хранением данных в `numpy.memmap`,
протоколом **stop-at-budget**, и сравнением:

- **LinUCB** (Disjoint)
- **Primal–Dual LinUCB (PD-LinUCB)** (динамический множитель/“теневая цена” ресурса)
- **CostNormUCB** (ratio/sub)
- **Context-free PD-BwK** (абляция “без контекста”)

---

## 0) Важное про артефакты статьи

В репозитории есть папка:

- `paper_artifacts/figures/` — финальные `.png` картинки, которые загружаются в Overleaf
- `paper_artifacts/tables/` — финальные `.tex` таблицы (например, `main_ci.tex`) для вставки в статью

Папка `results/` используется как **локальный scratch** для запусков и **не коммитится** (в `.gitignore`).

---

## 1) Быстрый старт

### 1.1 Окружение
Рекомендуется Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt