# Poker RL Go (starter)

Стартовый каркас для курсового проекта по RL для No-Limit Texas Hold'em cash games.

## Что внутри

- `docs/coursework_rl_nlhe_plan.md` — пошаговый план: выбор среды, baseline, обучение, self-play, оценка против сильных агентов.
- `docs/competitive_research_notes.md` — краткий обзор публичных подходов (DeepStack/Libratus/Pluribus/NFSP/Deep CFR/ReBeL) и что из этого применимо в проекте.
- `src/nlhe_action_mapper.py` — модуль маппинга и перечисления действий:
  - возврат legal действий `fold/call/check/raise`;
  - генерация рейзов от `0.01` банка до `1.00` банка (шаг настраиваемый);
  - явное добавление `all-in`;
  - ограничение конфигурации стола до `max 8` игроков.
- `src/holdem_equity.py` — расчет showdown equity (Monte Carlo) для NLHE, включая оценку 5/7-карточной силы руки, поддержка до 8 игроков.
- `src/rl_training_utils.py` — утилиты обучения: masked log-softmax, actor-critic loss, entropy bonus и шаг backpropagation с gradient clipping.
- `src/competitive_env_config.py` — конфигурация более соревновательной NLHE-среды (блайнды/анте/рейк/action abstraction/raise ladder).
- `src/self_play_league.py` — league-training утилиты: snapshot pool, Elo-апдейты, PFSP-сэмплинг оппонентов и промоут новых чекпоинтов.
- `src/rlcard_nlhe_env.py` — подключение реальной среды RLCard (`no-limit-holdem`) для сбора батчей из настоящих раздач и evaluation по рукам.
- `train.py` — обучение (backend `synthetic|rlcard`, GAE(λ), PPO minibatch updates (clip objective), checkpointing, league bookkeeping).
- `evaluate.py` — оценка чекпоинта в RLCard по средней выплате на руку.
- `evaluate_pool.py` — ранжирование пула чекпоинтов по средней выплате на руку.
- `configs/default.json` — воспроизводимый базовый конфиг для train/eval.
- `tests/test_nlhe_action_mapper.py` — unit-тесты для проверки поведения.
- `tests/test_holdem_equity.py` — тесты расчета эквити и hand evaluator.
- `tests/test_rl_training_utils.py` — тесты лосса и шага обратного распространения ошибки.
- `tests/test_competitive_env_config.py` — тесты для соревновательной конфигурации среды (players/rake/raise ladder).
- `tests/test_self_play_league.py` — тесты лиги self-play (Elo, PFSP, промоут, top-k).

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest
pytest -q
```

### Минимальный запуск обучения

```bash
python train.py --steps 200 --checkpoint-interval 50 --out-dir checkpoints --backend synthetic
```

### Запуск с реальной средой RLCard

```bash
pip install rlcard
python train.py --steps 200 --checkpoint-interval 50 --out-dir checkpoints_rlcard --backend rlcard
```

- `train.py` теперь поддерживает `--backend synthetic|rlcard`.
- Внутри используется GAE(λ) + нормализация advantages для более стабильного actor-critic апдейта.
- RLCard backend — это первый реальный bridge к соревновательной среде; следующим этапом стоит заменить random action sampling на policy-driven rollout collection.

Дальше см. roadmap в `docs/coursework_rl_nlhe_plan.md`.


### Оценка модели

```bash
python evaluate.py --checkpoint checkpoints_rlcard/policy_step_000200.pt --hands 500
```

Если не передавать `--checkpoint`, то считается baseline случайной легальной политики.


### Запуск через конфиг

```bash
python train.py --config configs/default.json
python evaluate.py --config configs/default.json
```


### Ранжирование пула чекпоинтов

```bash
python evaluate_pool.py --checkpoints-dir checkpoints_rlcard --hands 300
```
