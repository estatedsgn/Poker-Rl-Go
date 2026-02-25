# Poker RL Go (starter)

—тартовый каркас дл€ курсового проекта по RL дл€ No-Limit Texas Hold'em cash games.

## „то внутри

- `docs/coursework_rl_nlhe_plan.md` Ч пошаговый план: выбор среды, baseline, обучение, self-play, оценка против сильных агентов.
- `docs/competitive_research_notes.md` Ч краткий обзор публичных подходов (DeepStack/Libratus/Pluribus/NFSP/Deep CFR/ReBeL) и что из этого применимо в проекте.
- `src/nlhe_action_mapper.py` Ч модуль маппинга и перечислени€ действий:
  - возврат legal действий `fold/call/check/raise`;
  - генераци€ рейзов от `0.01` банка до `1.00` банка (шаг настраиваемый);
  - €вное добавление `all-in`;
  - ограничение конфигурации стола до `max 8` игроков.
- `src/holdem_equity.py` Ч расчет showdown equity (Monte Carlo) дл€ NLHE, включа€ оценку 5/7-карточной силы руки, поддержка до 8 игроков.
- `src/rl_training_utils.py` Ч утилиты обучени€: masked log-softmax, actor-critic loss, entropy bonus и шаг backpropagation с gradient clipping.
- `src/competitive_env_config.py` Ч конфигураци€ более соревновательной NLHE-среды (блайнды/анте/рейк/action abstraction/raise ladder).
- `src/self_play_league.py` Ч league-training утилиты: snapshot pool, Elo-апдейты, PFSP-сэмплинг оппонентов и промоут новых чекпоинтов.
- `src/rlcard_nlhe_env.py` Ч подключение реальной среды RLCard (`no-limit-holdem`) дл€ сбора батчей из насто€щих раздач и evaluation по рукам.
- `train.py` Ч обучение (backend `synthetic|rlcard`, GAE(?), checkpointing, league bookkeeping).
- `evaluate.py` Ч оценка чекпоинта в RLCard по средней выплате на руку.
- `evaluate_pool.py` Ч ранжирование пула чекпоинтов по средней выплате на руку.
- `configs/default.json` Ч воспроизводимый базовый конфиг дл€ train/eval.
- `tests/test_nlhe_action_mapper.py` Ч unit-тесты дл€ проверки поведени€.
- `tests/test_holdem_equity.py` Ч тесты расчета эквити и hand evaluator.
- `tests/test_rl_training_utils.py` Ч тесты лосса и шага обратного распространени€ ошибки.
- `tests/test_competitive_env_config.py` Ч тесты дл€ соревновательной конфигурации среды (players/rake/raise ladder).
- `tests/test_self_play_league.py` Ч тесты лиги self-play (Elo, PFSP, промоут, top-k).

## Ѕыстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest
pytest -q
```

### ћинимальный запуск обучени€

```bash
python train.py --steps 200 --checkpoint-interval 50 --out-dir checkpoints --backend synthetic
```

### «апуск с реальной средой RLCard

```bash
pip install rlcard
python train.py --steps 200 --checkpoint-interval 50 --out-dir checkpoints_rlcard --backend rlcard
```

- `train.py` теперь поддерживает `--backend synthetic|rlcard`.
- ¬нутри используетс€ GAE(?) + нормализаци€ advantages дл€ более стабильного actor-critic апдейта.
- RLCard backend Ч это первый реальный bridge к соревновательной среде; следующим этапом стоит заменить random action sampling на policy-driven rollout collection.

ƒальше см. roadmap в `docs/coursework_rl_nlhe_plan.md`.


### ќценка модели

```bash
python evaluate.py --checkpoint checkpoints_rlcard/policy_step_000200.pt --hands 500
```

≈сли не передавать `--checkpoint`, то считаетс€ baseline случайной легальной политики.


### «апуск через конфиг

```bash
python train.py --config configs/default.json
python evaluate.py --config configs/default.json
```


### –анжирование пула чекпоинтов

```bash
python evaluate_pool.py --checkpoints-dir checkpoints_rlcard --hands 300
```
